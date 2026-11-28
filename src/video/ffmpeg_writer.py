"""FFmpeg writer — drop-in cv2.VideoWriter dengan hardware encoder."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE: Optional[dict] = None


def _probe_encoder(ff: str, enc: str) -> tuple[bool, str]:
    """Encode 2 frame kecil untuk memastikan encoder benar-benar bisa dipakai
    (ffmpeg bisa saja mendaftar h264_nvenc walau driver NVENC terlalu lama)."""
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cmd = [ff, "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x64", "-r", "1",
               "-i", "-", *_codec_args(enc, 23), "-f", "null", "-"]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc = subprocess.run(
            cmd, input=(frame.tobytes() * 2),
            capture_output=True, timeout=20, creationflags=creationflags,
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return False, str(e)


def describe_encoder() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    info = {"ffmpeg": False, "ffmpeg_path": None, "encoder": None,
            "encoders_available": [], "working_encoders": [],
            "encoder_failures": {}, "error": None}
    ff = shutil.which("ffmpeg")
    if not ff:
        info["error"] = "ffmpeg tidak ditemukan di PATH."
        _CACHE = info
        return info
    info["ffmpeg"] = True
    info["ffmpeg_path"] = ff
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=5).stdout
        available = [e for e in ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")
                     if e in out]
        info["encoders_available"] = available
        for enc in available:
            ok, err = _probe_encoder(ff, enc)
            if ok:
                info["working_encoders"].append(enc)
            else:
                info["encoder_failures"][enc] = err
        if info["working_encoders"]:
            info["encoder"] = info["working_encoders"][0]
        else:
            info["error"] = "Tidak ada encoder H.264 yang berfungsi."
    except Exception as e:
        info["error"] = f"Gagal query ffmpeg: {e}"
    _CACHE = info
    return info


def _codec_args(enc: str, crf: int) -> list[str]:
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4",
                "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast",
                "-global_quality", str(crf), "-pix_fmt", "yuv420p"]
    if enc == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "speed",
                "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(crf), "-pix_fmt", "yuv420p"]


class FFmpegWriter:
    def __init__(self, path, width, height, fps=30, crf=23, prefer_encoder=None):
        info = describe_encoder()
        if not info["ffmpeg"]:
            raise RuntimeError(info["error"] or "ffmpeg tidak tersedia.")
        working = info.get("working_encoders") or info["encoders_available"]
        candidates = []
        if prefer_encoder and prefer_encoder in working:
            candidates.append(prefer_encoder)
        if info["encoder"]:
            candidates.append(info["encoder"])
        for enc in working:
            if enc not in candidates:
                candidates.append(enc)
        candidates = [e for e in candidates if e in info["encoders_available"]]
        if not candidates:
            raise RuntimeError(
                info.get("error")
                or f"Tidak ada encoder H.264 yang tersedia. Ada: {info['encoders_available']}"
            )

        self.path = str(path)
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.crf = int(crf)
        self._closed = False
        self.encoder = None
        self._p = None
        self.creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        )

        last_err = ""
        for enc in candidates:
            try:
                p = self._spawn(enc)
            except Exception as e:
                last_err = f"{enc}: {e}"
                continue
            if p.poll() is None:
                self._p = p
                self.encoder = enc
                break
            last_err = f"{enc}: {self._read_stderr(p)}"
        if self._p is None:
            raise RuntimeError(f"Tidak ada encoder H.264 yang bisa dipakai. {last_err}")

    def _spawn(self, enc: str):
        return subprocess.Popen(
            [describe_encoder()["ffmpeg_path"], "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
             "-i", "-", *_codec_args(enc, self.crf),
             "-movflags", "+faststart", self.path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=self.creationflags,
        )

    @staticmethod
    def _read_stderr(p) -> str:
        try:
            if p.stderr:
                return p.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return ""

    def isOpened(self):  # noqa
        return self._p is not None and not self._closed and self._p.poll() is None

    def write(self, frame):
        if self._closed:
            raise RuntimeError("writer sudah ditutup")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(f"dimensi {frame.shape[:2]} != writer ({self.height},{self.width})")
        try:
            self._p.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            err = self._read_stderr(self._p)
            raise RuntimeError(
                f"Encoder '{self.encoder}' gagal menulis frame: {e}. {err}"
            ) from e

    def release(self):
        if self._closed:
            return
        self._closed = True
        if self._p is None:
            return
        try:
            self._p.stdin.close()
        except Exception:
            pass
        rc = self._p.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exit {rc}: {self._read_stderr(self._p)}")