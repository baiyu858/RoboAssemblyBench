from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from roboassemblybench.datasets.media_codecs import (
    ChunkedDepthWriter,
    FFmpegVideoWriter,
    bitshuffle_uint16,
    bitunshuffle_uint16,
    depth_meters_to_uint16,
    read_depth_chunk,
)


def test_uint16_bitshuffle_round_trip_for_non_byte_aligned_shape():
    values = np.random.default_rng(7).integers(0, 65536, size=(13, 17), dtype=np.uint16)

    restored = bitunshuffle_uint16(bitshuffle_uint16(values), shape=values.shape)

    np.testing.assert_array_equal(restored, values)


def test_chunked_depth_round_trip_and_metadata(tmp_path: Path):
    pytest.importorskip('zstandard')
    frames = [
        np.asarray([[0.0, 1.2344, np.nan], [65.9, -1.0, 0.001]], dtype=np.float32),
        np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
    ]
    path = tmp_path / 'depth.u16.bshuf.zst'
    writer = ChunkedDepthWriter(path, shape=frames[0].shape, compression_level=3)
    for frame in frames:
        writer.write(frame)
    metadata = writer.close()

    assert metadata['dtype'] == 'uint16'
    assert metadata['depth_scale'] == 0.001
    assert metadata['filter'] == 'bitshuffle'
    assert metadata['chunking'] == 'frame'
    assert len(metadata['chunks']) == len(frames)
    for frame_index, frame in enumerate(frames):
        np.testing.assert_array_equal(read_depth_chunk(path, metadata, frame_index), depth_meters_to_uint16(frame))


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg is unavailable')
def test_ffmpeg_video_writer_uses_h264_crf(tmp_path: Path):
    path = tmp_path / 'rgb.mp4'
    writer = FFmpegVideoWriter(path, width=32, height=24, fps=10, codec='h264', crf=21)
    assert 'libx264' in writer.command
    assert writer.command[writer.command.index('-crf') + 1] == '21'
    for index in range(4):
        frame = np.full((24, 32, 3), index * 50, dtype=np.uint8)
        writer.write(frame)
    metadata = writer.close()

    assert path.is_file() and path.stat().st_size > 0
    assert metadata['codec'] == 'h264'
    assert metadata['encoding'] == 'crf'
    assert metadata['frame_count'] == 4
