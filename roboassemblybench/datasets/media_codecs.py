from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


DEPTH_MAGIC = b'RABDU16\0'
DEPTH_VERSION = 1
DEPTH_HEADER = struct.Struct('<8sI')
DEPTH_CHUNK_HEADER = struct.Struct('<III')
DEPTH_SCALE = 0.001


def depth_meters_to_uint16(depth_meters: np.ndarray) -> np.ndarray:
    """Quantize metric depth to millimeters; zero represents invalid depth."""

    depth = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    millimeters = np.zeros(depth.shape, dtype=np.uint16)
    if np.any(valid):
        quantized = np.rint(depth[valid] / DEPTH_SCALE)
        millimeters[valid] = np.clip(quantized, 1, np.iinfo(np.uint16).max).astype(np.uint16)
    return millimeters


def bitshuffle_uint16(values: np.ndarray) -> bytes:
    """Transpose uint16 sample bits into contiguous bit planes."""

    flattened = np.ascontiguousarray(values, dtype=np.uint16).reshape(-1)
    bytes_per_plane = (flattened.size + 7) // 8
    shuffled = np.empty((16, bytes_per_plane), dtype=np.uint8)
    for bit_index in range(16):
        bit_plane = ((flattened >> bit_index) & 1).astype(np.uint8, copy=False)
        shuffled[bit_index] = np.packbits(bit_plane, bitorder='little')
    return shuffled.tobytes(order='C')


def bitunshuffle_uint16(payload: bytes, *, shape: tuple[int, ...]) -> np.ndarray:
    sample_count = int(np.prod(shape, dtype=np.int64))
    bytes_per_plane = (sample_count + 7) // 8
    expected_size = 16 * bytes_per_plane
    if len(payload) != expected_size:
        raise ValueError(f'Invalid bitshuffled uint16 payload: {len(payload)} != {expected_size}.')
    planes = np.frombuffer(payload, dtype=np.uint8).reshape(16, bytes_per_plane)
    values = np.zeros(sample_count, dtype=np.uint16)
    for bit_index in range(16):
        bits = np.unpackbits(planes[bit_index], bitorder='little')[:sample_count]
        values |= bits.astype(np.uint16, copy=False) << bit_index
    return values.reshape(shape)


class ChunkedDepthWriter:
    """Write independently compressed, indexed uint16 depth frames."""

    def __init__(self, path: Path, *, shape: tuple[int, int], compression_level: int = 5):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError('Depth recording requires the zstandard Python package.') from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError(f'Depth shape must be two positive dimensions, got {self.shape}.')
        self.compression_level = int(compression_level)
        self._compressor = zstd.ZstdCompressor(level=self.compression_level)
        self._handle = self.path.open('wb')
        self._handle.write(DEPTH_HEADER.pack(DEPTH_MAGIC, DEPTH_VERSION))
        self._chunks: list[dict[str, int]] = []
        self._closed = False

    @property
    def count(self) -> int:
        return len(self._chunks)

    def write(self, depth_meters: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f'Depth stream is already closed: {self.path}.')
        depth = np.asarray(depth_meters)
        if tuple(depth.shape) != self.shape:
            raise ValueError(f'Depth frame changed shape from {self.shape} to {tuple(depth.shape)}.')
        quantized = depth_meters_to_uint16(depth)
        shuffled = bitshuffle_uint16(quantized)
        compressed = self._compressor.compress(shuffled)
        offset = self._handle.tell()
        self._handle.write(DEPTH_CHUNK_HEADER.pack(self.count, len(shuffled), len(compressed)))
        self._handle.write(compressed)
        self._chunks.append(
            {
                'frame_index': self.count,
                'offset': int(offset),
                'compressed_size': len(compressed),
                'uncompressed_size': len(shuffled),
            }
        )

    def close(self) -> dict[str, Any]:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True
        return {
            'path': str(self.path),
            'shape': list(self.shape),
            'dtype': 'uint16',
            'units': 'millimeters',
            'depth_scale': DEPTH_SCALE,
            'invalid_value': 0,
            'compression': 'zstd',
            'compression_level': self.compression_level,
            'filter': 'bitshuffle',
            'chunking': 'frame',
            'container': 'roboassemblybench_chunked_depth_v1',
            'count': self.count,
            'chunks': list(self._chunks),
        }


def read_depth_chunk(path: Path, metadata: dict[str, Any], frame_index: int) -> np.ndarray:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError('Depth decoding requires the zstandard Python package.') from exc

    chunks = metadata.get('chunks') or []
    if not 0 <= int(frame_index) < len(chunks):
        raise IndexError(frame_index)
    chunk = chunks[int(frame_index)]
    with Path(path).open('rb') as handle:
        magic, version = DEPTH_HEADER.unpack(handle.read(DEPTH_HEADER.size))
        if magic != DEPTH_MAGIC or version != DEPTH_VERSION:
            raise ValueError(f'Unsupported depth container in {path}.')
        handle.seek(int(chunk['offset']))
        stored_index, uncompressed_size, compressed_size = DEPTH_CHUNK_HEADER.unpack(
            handle.read(DEPTH_CHUNK_HEADER.size)
        )
        if stored_index != int(frame_index):
            raise ValueError(f'Depth chunk index mismatch: {stored_index} != {frame_index}.')
        compressed = handle.read(compressed_size)
    payload = zstd.ZstdDecompressor().decompress(compressed, max_output_size=uncompressed_size)
    return bitunshuffle_uint16(payload, shape=tuple(int(value) for value in metadata['shape']))


class FFmpegVideoWriter:
    """Stream RGB frames to an H.264/H.265 CRF-encoded MP4."""

    CODECS = {
        'h264': ('libx264', 'yuv420p'),
        'h265': ('libx265', 'yuv420p'),
        'hevc': ('libx265', 'yuv420p'),
    }

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        codec: str = 'h264',
        crf: int = 23,
        preset: str = 'veryfast',
        encoder_threads: int | None = None,
        ffmpeg_binary: str = 'ffmpeg',
    ):
        normalized_codec = str(codec).strip().lower()
        if normalized_codec not in self.CODECS:
            raise ValueError(f'Unsupported RGB codec {codec!r}; expected h264 or h265.')
        resolved_ffmpeg = shutil.which(ffmpeg_binary)
        if resolved_ffmpeg is None:
            raise RuntimeError(f'FFmpeg executable not found: {ffmpeg_binary}.')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(int(fps), 1)
        self.codec = 'h265' if normalized_codec in {'h265', 'hevc'} else 'h264'
        self.encoder, pixel_format = self.CODECS[normalized_codec]
        self.crf = int(crf)
        self.preset = str(preset)
        self.encoder_threads = max(
            int(encoder_threads if encoder_threads is not None else os.environ.get('RAB_FFMPEG_THREADS', '1')),
            1,
        )
        self.frame_count = 0
        self.log_path = self.path.with_suffix(f'{self.path.suffix}.ffmpeg.log')
        self._log_handle = self.log_path.open('wb')
        self.command = [
            resolved_ffmpeg,
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgb24',
            '-s:v',
            f'{self.width}x{self.height}',
            '-r',
            str(self.fps),
            '-i',
            '-',
            '-an',
            '-c:v',
            self.encoder,
            '-preset',
            self.preset,
            '-crf',
            str(self.crf),
            '-threads',
            str(self.encoder_threads),
            '-pix_fmt',
            pixel_format,
            '-movflags',
            '+faststart',
            str(self.path),
        ]
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_handle,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError(f'Video writer is already closed: {self.path}.')
        rgb = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if tuple(rgb.shape) != expected_shape:
            raise ValueError(f'RGB frame changed shape from {expected_shape} to {tuple(rgb.shape)}.')
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(np.ascontiguousarray(rgb).tobytes(order='C'))
        except BrokenPipeError as exc:
            self.close()
            raise RuntimeError(f'FFmpeg stopped while writing {self.path}; inspect {self.log_path}.') from exc
        self.frame_count += 1

    def close(self) -> dict[str, Any]:
        if not self._closed:
            if self._process.stdin is not None:
                self._process.stdin.close()
            returncode = self._process.wait()
            self._log_handle.close()
            self._closed = True
            if returncode != 0:
                detail = self.log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
                raise RuntimeError(f'FFmpeg failed for {self.path} with exit {returncode}: {detail}')
            if self.log_path.exists() and self.log_path.stat().st_size == 0:
                self.log_path.unlink()
        return {
            'path': str(self.path),
            'codec': self.codec,
            'encoder': self.encoder,
            'encoding': 'crf',
            'crf': self.crf,
            'preset': self.preset,
            'encoder_threads': self.encoder_threads,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
        }
