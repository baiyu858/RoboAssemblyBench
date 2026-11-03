"""Bounded asynchronous FFmpeg encoding, without per-frame PNG intermediates."""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np


class StreamingVideoWriter:
    """Keep every accepted frame; apply backpressure rather than dropping it.

    A small queue bounds host memory. Fragmented MP4 retains completed fragments
    if the simulation worker is interrupted. Encoding failure is never silent.
    """

    def __init__(self, path: Path, *, width: int, height: int, fps: float,
                 queue_size: int = 2, preset: str = 'veryfast', crf: int = 23):
        if width <= 0 or height <= 0 or width % 2 or height % 2 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Streaming video requires positive even dimensions and positive finite fps.')
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError('Video queue size must be a positive integer.')
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('FFmpeg is required for streaming video; use the PNG backend as a fallback.')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, float(fps)
        self.preset, self.crf = preset, crf
        self.frames_submitted = 0
        self.frames_written = 0
        self.max_queue_size = 0
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._log = self.path.with_suffix('.ffmpeg.log').open('wb')
        command = [executable, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                   '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                   '-an', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-threads', '2',
                   '-pix_fmt', 'yuv420p', '-g', str(max(1, round(fps))),
                   '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(self.path)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL, stderr=self._log)
        except Exception:
            self._log.close()
            raise
        self._thread = threading.Thread(target=self._encode, name='video-encoder', daemon=True)
        self._thread.start()

    def _encode(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._process.stdin.write(frame.tobytes())
                self.frames_written += 1
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self._process.stdin.close()
                code = self._process.wait(timeout=30)
                if code and self._error is None:
                    self._error = RuntimeError(f'FFmpeg exited with code {code}')
            except BaseException as exc:
                self._error = self._error or exc
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait()
            self._log.close()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f'Video encoding failed for {self.path}; see its .ffmpeg.log') from self._error
        if not self._thread.is_alive():
            raise RuntimeError(f'Video encoder stopped before finalization: {self.path}')

    def write(self, frame):
        if self._closed:
            raise RuntimeError('Cannot write to a finalized video.')
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError('Video frame must be uint8 RGB with the configured dimensions.')
        frame = np.array(frame, order='C', copy=True)  # Do not retain mutable sensor buffers.
        while True:
            self._check()
            try:
                self._queue.put(frame, timeout=.1)
                break
            except queue.Full:
                pass
        self.frames_submitted += 1
        self.max_queue_size = max(self.max_queue_size, self._queue.qsize())

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                while self._thread.is_alive():
                    self._check()
                    try:
                        self._queue.put(None, timeout=.1)
                        break
                    except queue.Full:
                        pass
            finally:
                self._thread.join(timeout=35)
                if self._thread.is_alive():
                    self._process.kill()
                    self._thread.join(timeout=5)
                    raise RuntimeError(f'Video encoder did not finish: {self.path}')
        if self._error is not None or self.frames_written != self.frames_submitted:
            raise RuntimeError(f'Incomplete video encoding for {self.path}') from self._error
        return {'encoder': 'ffmpeg_streaming_h264', 'frame_count': self.frames_written,
                'queue_capacity': self._queue.maxsize, 'max_queue_size': self.max_queue_size,
                'dropped_frames': 0, 'fragmented_mp4': True, 'preset': self.preset, 'crf': self.crf}
