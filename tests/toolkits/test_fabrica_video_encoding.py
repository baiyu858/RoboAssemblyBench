from pathlib import Path
from types import SimpleNamespace

from toolkits.factory_dual_franka_assembly import render_fabrica_official_motion_isaac as encoder


def test_encode_mp4_prefers_external_ffmpeg_when_available(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(2):
        (frames_dir / f"rgb_{index:05d}.png").write_bytes(b"not decoded by this test")

    output_path = tmp_path / "out.mp4"
    calls = []

    monkeypatch.setattr(encoder.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def fake_run(command, check, stdout, stderr):
        calls.append(command)
        output_path.write_bytes(b"mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(encoder, "subprocess", SimpleNamespace(run=fake_run, PIPE=object()), raising=False)

    def fail_get_writer(*args, **kwargs):
        raise AssertionError("imageio fallback should not be used when ffmpeg is available")

    monkeypatch.setattr(encoder.imageio, "get_writer", fail_get_writer)

    png_paths = encoder._encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=30)

    assert png_paths == sorted(str(path) for path in frames_dir.rglob("*.png"))
    assert calls
    assert calls[0][:5] == ["/usr/bin/ffmpeg", "-y", "-framerate", "30", "-i"]
    assert calls[0][-1] == str(output_path)
