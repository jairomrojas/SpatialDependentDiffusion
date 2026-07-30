"""FFmpeg codec selection shared by lightweight and Matplotlib renderers."""

from __future__ import annotations

import platform
import shutil
import subprocess


def _available_ffmpeg_encoders() -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for MP4 output. On macOS install it with: "
            "brew install ffmpeg"
        )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _videotoolbox_is_usable() -> bool:
    """The encoder can be listed yet unavailable in a remote/sandboxed session."""

    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "h264_videotoolbox",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    return probe.returncode == 0


def choose_codec(requested: str) -> str:
    if requested != "auto":
        return requested
    encoders = _available_ffmpeg_encoders()
    if (
        platform.system() == "Darwin"
        and "h264_videotoolbox" in encoders
        and _videotoolbox_is_usable()
    ):
        return "h264_videotoolbox"
    if "libx264" in encoders:
        return "libx264"
    return "h264"
