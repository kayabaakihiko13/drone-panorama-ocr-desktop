"""
Orchestrator Master Scheduler (TDD_Container_OCR_Windows_v1.9.md).
- In-process threading.Thread (Single-writer to SQLite).
- Thread-safe communication with PySide6 UI via queue.Queue.
- Orchestrates Stitching Adapter, ONNX Local Detector, and Halotec OCR API.
- Implements Zero Double-Detection: crops raw stitched panorama using Stage 4 JSON coordinates.
"""
from __future__ import annotations

import os
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import json
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import BASE_DIR, AppConfig
from src.database.db import Database
from src.ocr_client.client import HalotecOcrClient, OcrClientError
from src.recognition.detector import DetectionBox, OnnxDetector, describe_compute_plan
from src.stitching.adapter import StitchingAdapter
from src.utils.image_io import imread_unicode, imwrite_unicode
from src.utils.iso6346 import validate_iso6346, validate_iso_size_type
from src.orchestrator.annotation_motion import (
    _project_box_homography,
    box_iou,
    estimate_cumulative_motion,
    map_box,
    overlays_from_front_rear_tracks,
    overlays_from_stitching_tracks,
    render_overlay_video,
)


@dataclass
class OrchestratorEvent:
    type: str  # JOB_STARTED, PROGRESS, CONTAINER_PROCESSED, JOB_COMPLETED, JOB_FAILED, LOG
    job_id: int
    data: Dict[str, Any]


class ApiAuditLogger:
    """Thread-safe audit logger persisting API calls to json and emitting into unified job activity log."""

    def __init__(
        self,
        txt_path: Optional[Path],
        json_path: Path,
        evt_queue: queue.Queue,
        job_id: int,
        emit_fn: Optional[Callable[[OrchestratorEvent], None]] = None,
        write_txt: bool = True,
    ):
        self.txt_path = Path(txt_path) if txt_path else None
        self.json_path = Path(json_path)
        self.evt_queue = evt_queue
        self.emit_fn = emit_fn
        self.write_txt = write_txt and (self.txt_path is not None)
        self.job_id = job_id
        self._lock = threading.Lock()
        self.records: List[Dict[str, Any]] = []

        # Inisialisasi file log seketika saat job dimulai
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            if self.write_txt and self.txt_path is not None:
                self.txt_path.parent.mkdir(parents=True, exist_ok=True)
                if not self.txt_path.exists():
                    with open(self.txt_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"=== Halotec Container OCR API Audit Log - Job #{self.job_id} ===\n"
                            f"Dimulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"{'='*65}\n\n"
                        )
            if not self.json_path.exists():
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
        except Exception as e:
            print(f"[WARN] Inisialisasi api log gagal: {e}")

    def get_stats(self) -> Tuple[int, int]:
        """Returns (total_requests_sent, http_200_success_count)."""
        with self._lock:
            success_count = sum(1 for r in self.records if r.get("event") == "RESPONSE_RECEIVED" and r.get("status") == "SUCCESS")
            total_reqs = sum(1 for r in self.records if r.get("event") == "REQUEST_SENDING")
            return total_reqs, success_count

    def finalize(self) -> None:
        """Writes summary footer to txt_path upon job completion when standalone text log is enabled."""
        if not self.write_txt or self.txt_path is None:
            return
        total_reqs, success_count = self.get_stats()
        with self._lock:
            try:
                with open(self.txt_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n{'='*65}\n"
                        f"Ringkasan Sesi Job #{self.job_id}:\n"
                        f"Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Total Permintaan Dikirim: {total_reqs}\n"
                        f"Total Berhasil (HTTP 200): {success_count}\n"
                        f"{'='*65}\n"
                    )
            except Exception as e:
                print(f"[WARN] Finalisasi api log gagal: {e}")

    def on_api_event(self, event_data: Dict[str, Any]) -> None:
        with self._lock:
            evt_type = event_data.get("event")
            now_str = datetime.now().strftime("%H:%M:%S")
            ctx = event_data.get("context_info", "")
            det_type = event_data.get("detection_type", "")
            attempt = event_data.get("attempt", 1)

            if evt_type == "REQUEST_SENDING":
                kb = event_data.get("payload_kb", 0.0)
                url = event_data.get("url", "")
                gate = event_data.get("gate", "1")
                cam_pos = event_data.get("camera_position", "rear")
                line = (
                    f"[{now_str}] [KIRIM KE SERVER] {ctx} ({det_type}, cam={cam_pos}) -> "
                    f"POST {url} (gate={gate}, size={kb} KB, try #{attempt})"
                )
            elif evt_type == "RESPONSE_RECEIVED":
                status = event_data.get("status")
                code = event_data.get("status_code", 0)
                ms = event_data.get("duration_ms", 0.0)
                msg = event_data.get("server_message", "")

                if status == "SUCCESS":
                    text = event_data.get("text", "")
                    conf = event_data.get("confidence", 0.0)
                    line = (
                        f"[{now_str}] [SERVER OK] {ctx} ({det_type}) <- HTTP {code} ({ms}ms) | "
                        f"Text: '{text}' (conf: {conf:.2f}) | Msg: {msg}"
                    )
                elif status == "RATE_LIMIT":
                    line = (
                        f"[{now_str}] [SERVER RATE LIMIT] {ctx} ({det_type}) <- HTTP {code} ({ms}ms) | "
                        f"Pesan Server: '{msg}'. Menjalankan backoff retry #{attempt}..."
                    )
                elif status == "UNAUTHORIZED_REFRESHING":
                    line = f"[{now_str}] [SERVER AUTH] {ctx} <- HTTP 401 (Token expired). Memperbarui JWT..."
                elif status == "UNAUTHORIZED":
                    line = f"[{now_str}] [SERVER AUTH ERROR] {ctx} <- HTTP 401. Token tidak valid atau login gagal."
                elif status == "SERVER_ERROR":
                    line = f"[{now_str}] [SERVER ERROR] {ctx} <- HTTP {code} ({ms}ms): {msg}"
                elif status == "NETWORK_ERROR":
                    line = f"[{now_str}] [SERVER NETWORK ERROR] {ctx} <- Koneksi error: {msg}"
                else:
                    line = f"[{now_str}] [SERVER RESP] {ctx} <- HTTP {code} ({ms}ms): {msg}"
            else:
                line = f"[{now_str}] [API] {event_data}"

            # 1. UI event queue (also mirrors into unified job_activity.log when emit_fn is bound)
            evt = OrchestratorEvent(
                type="LOG",
                job_id=self.job_id,
                data={"log": line},
            )
            if self.emit_fn is not None:
                self.emit_fn(evt)
            else:
                self.evt_queue.put(evt)

            # 2. Append to standalone txt log only when explicitly enabled
            if self.write_txt and self.txt_path is not None:
                try:
                    with open(self.txt_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception as e:
                    print(f"[WARN] Gagal menulis api_requests.txt: {e}")

            # 3. Append to api_requests.json
            self.records.append(event_data)
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.records, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Gagal menulis api_requests.json: {e}")


class Orchestrator(threading.Thread):
    """Master Scheduler and Single-Writer for Container OCR Desktop."""

    def __init__(
        self,
        command_queue: queue.Queue,
        event_queue: queue.Queue,
        config: Optional[AppConfig] = None,
        db: Optional[Database] = None,
    ):
        super().__init__(name="OrchestratorThread", daemon=True)
        self.cmd_queue = command_queue
        self.evt_queue = event_queue
        self.config = config or AppConfig.load()
        self.db = db or Database()
        self.ocr_client = HalotecOcrClient(
            base_url=self.config.api.url,
            gate=self.config.api.gate,
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
        )
        self.stitching_adapter = StitchingAdapter(self.config)
        self._detector: Optional[OnnxDetector] = None
        self._running = True
        self._current_job_id: Optional[int] = None
        self._cancel_requested = threading.Event()
        self._active_log_path: Optional[Path] = None
        self._active_log_lock = threading.Lock()
        self._active_job_start: Optional[datetime] = None
        self._active_audit_logger: Optional[ApiAuditLogger] = None

    @staticmethod
    def _log_activity(log_path: Path, message: str) -> None:
        """Helper to append timestamped activity line to unified job_activity.log."""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{now_str}] {message}\n")
        except Exception:
            pass

    @staticmethod
    def _format_duration(seconds: float) -> str:
        secs = max(0, int(round(seconds)))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}j {m:02d}m {s:02d}d"
        if m > 0:
            return f"{m}m {s:02d}d"
        return f"{s}d"

    def _write_log_raw(self, text: str) -> None:
        log_p = self._active_log_path
        if log_p is None:
            return
        try:
            with self._active_log_lock:
                log_p.parent.mkdir(parents=True, exist_ok=True)
                with open(log_p, "a", encoding="utf-8") as f:
                    f.write(text)
        except Exception:
            pass

    def _append_unified_log_line(self, line: str, ts: Optional[datetime] = None) -> None:
        now_dt = ts or datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        full_ts = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        # Upgrade [HH:MM:SS] prefix from ApiAuditLogger to full [YYYY-MM-DD HH:MM:SS]
        if len(line) >= 10 and line[0] == "[" and line[3] == ":" and line[6] == ":" and line[9] == "]":
            formatted = f"[{date_str} {line[1:]}\n"
        else:
            formatted = f"[{full_ts}] {line}\n"
        self._write_log_raw(formatted)

    def _emit_event(self, evt: OrchestratorEvent) -> None:
        """Persists UI activity stream into unified job_activity.log and emits event to UI queue."""
        if self._active_log_path is not None:
            now_dt = datetime.now()
            if evt.type == "JOB_STARTED":
                d = evt.data or {}
                det_info = d.get("compute_summary", getattr(self.config.stitching, "device", "cpu"))
                self._write_log_raw(
                    f"{'=' * 72}\n"
                    f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] === MULAI JOB #{evt.job_id}: {d.get('name', 'Job')} ===\n"
                    f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] Mode: {d.get('mode')}, Sensor: {d.get('camera_position')}, Komputasi: {det_info}\n"
                    f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] Video: {d.get('video_path', '')}\n"
                    f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] Folder Sesi: {d.get('session_dir', '')}\n"
                    f"{'=' * 72}\n"
                )
            elif evt.type == "PROGRESS":
                d = evt.data or {}
                pct = int(d.get("progress", 0.0) * 100)
                log_entry = d.get("log")
                msg = d.get("message", "")
                if log_entry:
                    self._append_unified_log_line(str(log_entry), now_dt)
                elif msg:
                    if not str(msg).startswith("Tahap 1/4: Tracking frame") or pct % 10 == 0:
                        self._append_unified_log_line(f"[{pct:3d}%] {msg}", now_dt)
            elif evt.type == "LOG":
                d = evt.data or {}
                log_entry = d.get("log") or d.get("message")
                if log_entry:
                    self._append_unified_log_line(str(log_entry), now_dt)
            elif evt.type in ("JOB_COMPLETED", "JOB_FAILED", "JOB_CANCELLED"):
                d = evt.data or {}
                if evt.type == "JOB_COMPLETED":
                    tag_line = f"[SUKSES] {d.get('message', 'Job berhasil diselesaikan')}"
                    status_lbl = "COMPLETED"
                elif evt.type == "JOB_CANCELLED":
                    tag_line = f"[DIBATALKAN] {d.get('message', 'Proses dibatalkan oleh pengguna.')}"
                    status_lbl = "CANCELLED"
                else:
                    tag_line = f"[GAGAL] {d.get('error', 'Error tidak diketahui')}"
                    status_lbl = "FAILED"
                self._append_unified_log_line(tag_line, now_dt)

                start_dt = self._active_job_start or now_dt
                dur_str = self._format_duration((now_dt - start_dt).total_seconds())
                total_c, valid_c = 0, 0
                out_vid_str = "-"
                try:
                    views = self.db.get_full_container_view(evt.job_id)
                    total_c = len(views)
                    valid_c = sum(1 for c in views if c.get("detections", {}).get("container_number", {}).get("is_valid_iso"))
                    jobs_list = [j for j in self.db.get_all_jobs() if j["id"] == evt.job_id]
                    if jobs_list and jobs_list[0].get("output_video_path"):
                        out_vid_str = str(jobs_list[0]["output_video_path"])
                except Exception:
                    pass
                req_tot, req_ok = (0, 0)
                if self._active_audit_logger is not None:
                    req_tot, req_ok = self._active_audit_logger.get_stats()

                self._write_log_raw(
                    f"\n{'=' * 72}\n"
                    f"=== RINGKASAN EVALUASI JOB #{evt.job_id} ({status_lbl}) ===\n"
                    f"Mulai            : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Selesai          : {now_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Durasi Total     : {dur_str}\n"
                    f"Kontainer        : {total_c} terdeteksi | Valid ISO 6346: {valid_c}/{total_c}\n"
                    f"OCR Cloud Service: {req_ok}/{req_tot} permintaan berhasil (HTTP 200)\n"
                    f"Video Anotasi    : {out_vid_str}\n"
                    f"{'=' * 72}\n"
                )
            elif evt.type == "MANUAL_OVERRIDE_SAVED":
                d = evt.data or {}
                self._append_unified_log_line(
                    f"[MANUAL OVERRIDE] det_id={d.get('detection_id')} -> '{d.get('text')}' (ISO valid={d.get('is_valid')})",
                    now_dt,
                )

        self.evt_queue.put(evt)

    def cancel_current_job(self) -> None:
        """Immediately signals cancellation for the running job and terminates stitching subprocess."""
        self._cancel_requested.set()
        self.stitching_adapter.terminate()

    def _get_detector(self) -> OnnxDetector:
        req_dev = str(getattr(self.config.stitching, "device", "cpu") or "cpu").strip().lower()
        if self._detector is None or getattr(self._detector, "requested_device", "cpu") != req_dev:
            self._detector = OnnxDetector(
                self.config.detection.resolved_model_path(),
                device=req_dev,
            )
        return self._detector

    def stop(self) -> None:
        """Signals the orchestrator thread to terminate gracefully."""
        self._running = False
        self.stitching_adapter.terminate()
        self.cmd_queue.put({"action": "STOP"})

    def run(self) -> None:
        if self.config.auto_resume_interrupted:
            recovered = self.db.mark_interrupted_jobs()
            if recovered > 0:
                self._emit_event(
                    OrchestratorEvent(
                        type="LOG",
                        job_id=0,
                        data={"message": f"Ditemukan {recovered} job terputus (status diubah ke INTERRUPTED)"},
                    )
                )

        while self._running:
            try:
                cmd = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            action = cmd.get("action")
            try:
                if action == "STOP":
                    break
                elif action == "START_JOB":
                    self._handle_start_job(cmd)
                elif action == "CANCEL_JOB":
                    self.cancel_current_job()
                elif action == "MANUAL_OVERRIDE":
                    self._handle_manual_override(cmd)
                elif action == "RE_OCR_CROP":
                    self._handle_re_ocr_crop(cmd)
                elif action == "UPDATE_CONFIG":
                    self._handle_update_config(cmd)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                try:
                    self._emit_event(
                        OrchestratorEvent(
                            type="LOG",
                            job_id=self._current_job_id or 0,
                            data={"log": f"[ORCHESTRATOR ERROR] action={action}: {e}\n{tb}"},
                        )
                    )
                except Exception:
                    print(f"[FATAL] Gagal emit error: {e}\n{tb}")
            finally:
                try:
                    self.cmd_queue.task_done()
                except Exception:
                    pass

    def _handle_start_job(self, cmd: Dict[str, Any]) -> None:
        name = cmd.get("name", "Job")
        video_path = cmd.get("video_path", "")
        mode = cmd.get("mode", "LEFT_RIGHT")
        camera_position = cmd.get("camera_position")
        if not camera_position:
            if mode == "LEFT_RIGHT":
                camera_position = "left"
            elif mode == "TOP":
                camera_position = "top"
            else:
                camera_position = "rear"

        job_id = self.db.create_job(name=name, video_path=video_path, mode=mode)
        self._current_job_id = job_id
        self._cancel_requested.clear()
        self.db.update_job_status(job_id, "PROCESSING")

        job_start_dt = datetime.now()
        video_stem = Path(video_path).stem or f"job_{job_id}"
        job_output_dir = self.config.paths.resolved_output_dir() / f"{job_start_dt.strftime('%Y%m%d_%H%M%S')}_{video_stem}"
        (job_output_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._active_log_path = job_output_dir / "logs" / "job_activity.log"
        self._active_job_start = job_start_dt
        self._active_audit_logger = None

        req_dev = str(getattr(self.config.stitching, "device", "cpu") or "cpu").upper()
        plan = describe_compute_plan(req_dev)
        try:
            det_inst = self._get_detector()
            onnx_act = getattr(det_inst, "active_provider", plan["onnx_provider"])
        except Exception:
            onnx_act = plan["onnx_provider"]
        compute_summary = (
            f"Diminta={req_dev}, Tracking={plan['track_provider']}, "
            f"Stitching={plan['stitch_device']}, ONNX={onnx_act} [torch={plan['torch_version']}]"
        )

        self._emit_event(
            OrchestratorEvent(
                type="JOB_STARTED",
                job_id=job_id,
                data={
                    "name": name,
                    "video_path": video_path,
                    "mode": mode,
                    "camera_position": camera_position,
                    "compute_summary": compute_summary,
                    "session_dir": str(job_output_dir),
                },
            )
        )

        try:
            # Disk space preflight check (> 2 GB required)
            data_dir = self.config.paths.resolved_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(data_dir).free
            if free_bytes < 2 * 1024 * 1024 * 1024:
                raise RuntimeError(
                    f"Kapasitas ruang disk tidak mencukupi ({free_bytes / (1024**3):.2f} GB tersedia). "
                    "Dibutuhkan minimal 2.0 GB ruang kosong untuk pemrosesan video dan citra."
                )

            if mode in ("LEFT_RIGHT", "TOP"):
                self._process_stitching_job(job_id, video_path, mode, camera_position, job_output_dir=job_output_dir)
            elif mode == "FRONT_REAR":
                self._process_front_rear_job(job_id, video_path, camera_position, job_output_dir=job_output_dir)
            else:
                raise ValueError(f"Unknown processing mode: {mode}")

            self.db.update_job_status(job_id, "COMPLETED")
            self._emit_event(
                OrchestratorEvent(
                    type="JOB_COMPLETED",
                    job_id=job_id,
                    data={"message": "Job berhasil diselesaikan"},
                )
            )
        except Exception as e:
            error_msg = str(e)
            is_cancelled = self._cancel_requested.is_set() or "dibatalkan" in error_msg.lower()
            if is_cancelled:
                self.db.update_job_status(job_id, "INTERRUPTED", error_message="Dibatalkan oleh pengguna")
                self.db.log_audit("WARNING", f"Job {job_id} dibatalkan oleh pengguna", job_id=job_id)
                self._emit_event(
                    OrchestratorEvent(
                        type="JOB_CANCELLED",
                        job_id=job_id,
                        data={"message": "Proses dibatalkan oleh pengguna"},
                    )
                )
            else:
                self.db.update_job_status(job_id, "FAILED", error_message=error_msg)
                self.db.log_audit("ERROR", f"Job {job_id} gagal: {error_msg}", job_id=job_id)
                self._emit_event(
                    OrchestratorEvent(
                        type="JOB_FAILED",
                        job_id=job_id,
                        data={"error": error_msg},
                    )
                )
        finally:
            self._current_job_id = None
            self._active_log_path = None
            self._active_job_start = None
            self._active_audit_logger = None

    def _process_stitching_job(
        self,
        job_id: int,
        video_path: str,
        mode: str,
        camera_position: str = "left",
        job_output_dir: Optional[Path] = None,
    ) -> None:
        # 1. Output directory with precise timestamp: outputs/YYYYMMDD_HHMMSS_<video_stem>
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(video_path).stem
        if job_output_dir is None:
            job_output_dir = self.config.paths.resolved_output_dir() / f"{timestamp_str}_{video_stem}"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        temp_crop_dir = job_output_dir / ".temp_crops"
        temp_crop_dir.mkdir(parents=True, exist_ok=True)
        images_dir = job_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        job_logs_dir = job_output_dir / "logs"
        job_logs_dir.mkdir(parents=True, exist_ok=True)
        api_log_json_path = job_logs_dir / "api_requests.json"
        activity_log_path = job_logs_dir / "job_activity.log"
        if self._active_log_path is None:
            self._active_log_path = activity_log_path
            self._active_job_start = datetime.now()
            self._log_activity(activity_log_path, f"Job #{job_id} dimulai (Mode: {mode}, Posisi: {camera_position}). Video: {video_path}")
        audit_logger = ApiAuditLogger(
            txt_path=activity_log_path,
            json_path=api_log_json_path,
            evt_queue=self.evt_queue,
            job_id=job_id,
            emit_fn=self._emit_event,
            write_txt=False,
        )
        self._active_audit_logger = audit_logger

        tracks_received = threading.Event()
        tracks_data_holder: List[Dict[str, Any]] = []
        stitching_completed = threading.Event()
        stitching_error: List[Exception] = []

        def on_stitching_progress(msg: str, pct: float) -> None:
            if self._cancel_requested.is_set():
                raise RuntimeError("Proses dibatalkan oleh pengguna")
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={"stage": "STITCHING", "message": msg, "progress": pct},
                )
            )

        def on_tracks_ready(tracks_meta: List[Dict[str, Any]]) -> None:
            tracks_data_holder.extend(tracks_meta)
            tracks_received.set()
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "TRACKING_DONE",
                        "message": f"Tracking selesai: {len(tracks_meta)} kontainer teridentifikasi. Memulai OCR paralel dari frame 4K...",
                        "progress": 0.35,
                        "log": f"Ditemukan {len(tracks_meta)} kontainer valid. Pipeline OCR paralel 4K diaktifkan.",
                    },
                )
            )

        def on_panorama_ready(pano_evt: Dict[str, Any]) -> None:
            route_order = pano_evt.get("route_order", 0)
            track_id = pano_evt.get("track_id", 0)
            img_p = pano_evt.get("image_path", "")
            ann_p = pano_evt.get("annotated_image_path", "")
            self.db.update_container_image(
                job_id=job_id,
                route_order=route_order,
                image_path=img_p,
                annotated_image_path=ann_p,
                status="stable",
            )
            self._emit_event(
                OrchestratorEvent(
                    type="CONTAINER_PROCESSED",
                    job_id=job_id,
                    data={"route_order": route_order, "track_id": track_id, "stage": "STITCHING_READY"},
                )
            )
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "STITCHING",
                        "message": f"Panorama R{route_order:03d} (Track {track_id}) selesai dijahit",
                        "progress": 0.70,
                        "log": f"[STITCHING] R{route_order:03d} (Track {track_id}): Citra panorama selesai dijahit & terpasang.",
                    },
                )
            )

        self._emit_event(
            OrchestratorEvent(
                type="PROGRESS",
                job_id=job_id,
                data={
                    "stage": "INIT",
                    "message": "Tahap 1/4: Menyiapkan tracking video...",
                    "progress": 0.02,
                    "log": f"Folder output sesi: {job_output_dir.name}",
                },
            )
        )

        returned_json_labels: List[Path] = []

        def run_stitching_worker():
            try:
                res = self.stitching_adapter.run_stitching(
                    video_path=video_path,
                    camera_position=mode,
                    output_dir=job_output_dir,
                    progress_callback=on_stitching_progress,
                    on_tracks_ready=on_tracks_ready,
                    on_panorama_ready=on_panorama_ready,
                    is_cancelled_callback=self._cancel_requested.is_set,
                )
                if res:
                    returned_json_labels.extend(res)
            except Exception as e:
                stitching_error.append(e)
            finally:
                tracks_received.set()
                stitching_completed.set()

        stitch_thread = threading.Thread(target=run_stitching_worker, name="StitchingWorkerThread", daemon=True)
        stitch_thread.start()

        # Tunggu stitching dan deteksi teks panorama selesai sepenuhnya
        while stitch_thread.is_alive():
            if self._cancel_requested.is_set():
                self.stitching_adapter.terminate()
                stitch_thread.join(timeout=2.0)
                raise RuntimeError("Proses dibatalkan oleh pengguna")
            time.sleep(0.2)

        if stitching_error:
            raise stitching_error[0]

        # Kumpulkan semua dokumen label JSON Stage 4
        if not returned_json_labels:
            labels_folder = job_output_dir / mode / "labels"
            if not labels_folder.is_dir():
                labels_folder = job_output_dir / "labels"
            if labels_folder.is_dir():
                returned_json_labels.extend(sorted(labels_folder.glob("*.json")))

        tracks_data_holder = []
        for jf in returned_json_labels:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                reg_str = doc.get("region_id", "")
                r_id = int(reg_str.split("R")[-1]) if "R" in reg_str else 1
                img_rel = doc.get("image_path", "")
                raw_img_p = (BASE_DIR / img_rel).resolve() if not Path(img_rel).is_absolute() else Path(img_rel)
                ann_img_p = str(raw_img_p).replace(".jpg", "_annotated.jpg")
                if not Path(ann_img_p).exists():
                    ann_img_p = str(raw_img_p)

                tracks_data_holder.append({
                    "route_order": r_id,
                    "track_id": int(doc.get("container_id", r_id)),
                    "pass_frame": doc.get("pass_frame"),
                    "container_video_bbox": doc.get("container_video_bbox"),
                    "panorama_container_bbox": doc.get("panorama_container_bbox"),
                    "video_frame_range": doc.get("video_frame_range"),
                    "frame_registrations": doc.get("frame_registrations", []),
                    "image_path": str(raw_img_p),
                    "annotated_image_path": ann_img_p,
                    "detections": doc.get("detections", {}),
                    "panorama_coordinates": doc.get("panorama_coordinates", {}),
                })
            except Exception as err:
                print(f"[WARN] Gagal membaca label {jf}: {err}")

        total_regions = len(tracks_data_holder)
        self.db.update_job_status(job_id, "PROCESSING", total_regions=total_regions)

        # 2. Pengenalan ROI Teks (Prioritas 4K Native + Fallback Panorama) & Pengiriman ke Halotec OCR API
        detector = self._get_detector()
        log_cb = audit_logger.on_api_event if audit_logger else None
        cap_4k = cv2.VideoCapture(str(video_path))
        vid_w = int(cap_4k.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if cap_4k.isOpened() else 0
        vid_h = int(cap_4k.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if cap_4k.isOpened() else 0
        vid_total = int(cap_4k.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap_4k.isOpened() else 0
        preferred_top_rot: List[str] = ["CW", "CCW", "RAW"]
        text_thresh = min(0.18, float(self.config.detection.threshold_text))

        def _run_ocr_num_crop(
            src_crop: Any,
            route_idx: int,
            c_type: str,
            cam_pos: str,
            src_label: str,
            allow_rotate: bool,
            best_state: Dict[str, Any],
        ) -> bool:
            """Evaluate candidate number crop via Halotec OCR API without overwriting final _num.jpg on failures."""
            c_h, c_w = src_crop.shape[:2]
            if c_h < 6 or c_w < 6:
                return False
            crop_scaled = src_crop
            if min(c_h, c_w) < 40 or max(c_h, c_w) < 150:
                crop_scaled = cv2.resize(src_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            if allow_rotate:
                # TOP roofs may be horizontal or rotated CW/CCW 90 deg
                is_vert = (c_h > c_w * 1.1)
                rot_order = (
                    list(preferred_top_rot)
                    if is_vert
                    else ["RAW"] + [r for r in preferred_top_rot if r != "RAW"]
                )
            else:
                # LEFT_RIGHT corner posts are already upright — NEVER rotate horizontally
                rot_order = ["RAW"]

            rot_imgs = {
                "RAW": crop_scaled,
                "CW": cv2.rotate(crop_scaled, cv2.ROTATE_90_CLOCKWISE),
                "CCW": cv2.rotate(crop_scaled, cv2.ROTATE_90_COUNTERCLOCKWISE),
            }
            cand_tmp_p = temp_crop_dir / f"job{job_id}_reg{route_idx:03d}_cand.jpg"
            for r_name in rot_order:
                cand_img = rot_imgs[r_name]
                imwrite_unicode(cand_tmp_p, cand_img)
                try:
                    txt_res, conf_res = self.ocr_client.recognize_image_file(
                        cand_tmp_p,
                        detection_type="container_number",
                        camera_position=cam_pos,
                        req_id=job_id,
                        log_callback=log_cb,
                        context_info=f"Kontainer R{route_idx:03d} ({c_type}, {src_label}-{r_name})",
                    )
                except Exception:
                    txt_res, conf_res = "", 0.0

                iso_res = validate_iso6346(txt_res)
                if iso_res.is_valid:
                    best_state["clean_text"] = iso_res.clean_number
                    best_state["raw_text"] = txt_res
                    best_state["is_valid"] = True
                    best_state["best_conf"] = conf_res
                    best_state["best_img"] = cand_img.copy()
                    if allow_rotate and r_name in preferred_top_rot:
                        preferred_top_rot.remove(r_name)
                        preferred_top_rot.insert(0, r_name)
                    return True
                elif txt_res and (not best_state["clean_text"] or len(txt_res) > len(best_state["clean_text"])):
                    best_state["clean_text"] = iso_res.clean_number if iso_res.clean_number else txt_res
                    best_state["raw_text"] = txt_res
                    best_state["best_conf"] = conf_res
                    best_state["best_img"] = cand_img.copy()
            return False

        try:
            for idx, track in enumerate(tracks_data_holder, start=1):
                if self._cancel_requested.is_set():
                    raise RuntimeError("Proses dibatalkan oleh pengguna")
                route_order = int(track.get("route_order", idx))
                container_id = int(track.get("track_id", idx))
                init_img_p = track.get("image_path", "")
                pos = track.get("panorama_coordinates", {})
                detected_c_type = "TOP" if mode == "TOP" else "LEFT_RIGHT"
                actual_cam_pos = "top" if detected_c_type == "TOP" else ("rear" if detected_c_type == "FRONT_REAR" else camera_position)
                effective_cam_pos = "rear" if detected_c_type in ("TOP", "FRONT_REAR") else actual_cam_pos

                raw_cv_img = imread_unicode(Path(init_img_p)) if (init_img_p and Path(init_img_p).is_file()) else None
                if raw_cv_img is None:
                    # Coba fallback path jika folder sudah dipindahkan ke .stitching
                    alt_p = Path(str(init_img_p).replace(f"/{mode}/", "/.stitching/").replace(f"\\{mode}\\", "\\.stitching\\"))
                    if alt_p.is_file():
                        raw_cv_img = imread_unicode(alt_p)
                if raw_cv_img is None:
                    continue

                cc_h, cc_w = raw_cv_img.shape[:2]
                track["pano_w"] = cc_w
                track["pano_h"] = cc_h

                # 1. Kumpulkan kandidat bounding box teks (dari JSON deteksi panorama dan/atau deteksi ONNX lokal)
                detections_map = track.get("detections", {})
                c_num_det = detections_map.get("container_number")
                iso_det = detections_map.get("iso_code")

                candidate_num_boxes: List[Tuple[int, int, int, int, float]] = []
                candidate_iso_boxes: List[Tuple[int, int, int, int, float]] = []
                nb_pano_xywh: Optional[Tuple[float, float, float, float]] = None
                iso_pano_xywh: Optional[Tuple[float, float, float, float]] = None

                if c_num_det and isinstance(c_num_det.get("bbox"), dict) and c_num_det["bbox"].get("width", 0) > 0:
                    bb = c_num_det["bbox"]
                    nb_pano_xywh = (float(bb["x"]), float(bb["y"]), float(bb["width"]), float(bb["height"]))
                    candidate_num_boxes.append((int(bb["x"]), int(bb["y"]), int(bb["x"] + bb["width"]), int(bb["y"] + bb["height"]), float(c_num_det.get("confidence", 0.70))))
                if iso_det and isinstance(iso_det.get("bbox"), dict) and iso_det["bbox"].get("width", 0) > 0:
                    bb = iso_det["bbox"]
                    iso_pano_xywh = (float(bb["x"]), float(bb["y"]), float(bb["width"]), float(bb["height"]))
                    candidate_iso_boxes.append((int(bb["x"]), int(bb["y"]), int(bb["x"] + bb["width"]), int(bb["y"] + bb["height"]), float(iso_det.get("confidence", 0.70))))

                # Jika belum ada deteksi pada label JSON, jalankan detektor ONNX langsung pada panorama
                if not candidate_num_boxes:
                    raw_boxes = detector.detect(
                        raw_cv_img,
                        conf_thresh=min(0.20, float(self.config.detection.threshold_text)),
                        imgsz=self.config.detection.imgsz_text_detection,
                        classes=[1, 2],
                    )
                    for b in raw_boxes:
                        if b.class_id == 2:
                            candidate_num_boxes.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(b.confidence)))
                        elif b.class_id == 1:
                            candidate_iso_boxes.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(b.confidence)))

                candidate_num_boxes.sort(key=lambda x: x[4], reverse=True)
                candidate_iso_boxes.sort(key=lambda x: x[4], reverse=True)

                # 2. Proses OCR Prioritas 4K Asli dari Video via frame_registrations (+ fallback panorama)
                ocr_state: Dict[str, Any] = {
                    "clean_text": "",
                    "raw_text": "",
                    "is_valid": False,
                    "best_conf": 0.0,
                    "best_img": None,
                }
                best_num_box: Optional[Tuple[int, int, int, int]] = (
                    (candidate_num_boxes[0][0], candidate_num_boxes[0][1], candidate_num_boxes[0][2], candidate_num_boxes[0][3])
                    if candidate_num_boxes else None
                )
                num_crop_path: Optional[Path] = None
                best_4k_frame: Optional[int] = None
                best_4k_num_box: Optional[Tuple[int, int, int, int]] = None
                best_4k_cont_box: Optional[Tuple[float, float, float, float]] = None
                best_4k_iso_crop: Optional[Any] = None
                best_4k_iso_box_pano: Optional[Tuple[int, int, int, int]] = None

                reg_list = track.get("frame_registrations") or []
                reg_by_frame: Dict[int, Dict[str, Any]] = {}
                for r_item in reg_list:
                    try:
                        reg_by_frame[int(r_item["frame"])] = r_item
                    except Exception:
                        pass

                cand_4k_frames: List[int] = []
                if cap_4k.isOpened() and vid_w > 0 and vid_h > 0:
                    scored_regs: List[Tuple[float, int]] = []
                    for rf, r_item in reg_by_frame.items():
                        H = np.asarray(r_item.get("H_pano_to_vid", []), dtype=np.float64)
                        db = r_item.get("det_bbox")
                        cx_score = 1.0
                        if H.shape == (3, 3) and nb_pano_xywh is not None:
                            pb = _project_box_homography(nb_pano_xywh, H)
                            if pb is not None:
                                ncx = (pb[0] + pb[2]) / 2.0
                                ncy = (pb[1] + pb[3]) / 2.0
                                if 0.06 * vid_w <= ncx <= 0.94 * vid_w and 0.04 * vid_h <= ncy <= 0.96 * vid_h:
                                    cx_score = abs(ncx - vid_w * 0.5) / float(vid_w)
                                else:
                                    cx_score = 2.0 + abs(ncx - vid_w * 0.5) / float(vid_w)
                        elif db and len(db) >= 4:
                            dcx = (float(db[0]) + float(db[2])) / 2.0
                            cx_score = 0.5 + abs(dcx - vid_w * 0.5) / float(vid_w)
                        scored_regs.append((cx_score, rf))

                    scored_regs.sort(key=lambda x: x[0])
                    for _, rf in scored_regs:
                        if all(abs(rf - kept_f) >= 10 for kept_f in cand_4k_frames):
                            cand_4k_frames.append(rf)
                        if len(cand_4k_frames) >= 3:
                            break

                    # Add early/mid/late registered frames so opposite container ends (e.g. TOP dual-end text) are covered
                    reg_fs_sorted = sorted(reg_by_frame.keys())
                    if reg_fs_sorted:
                        span_picks = [
                            reg_fs_sorted[len(reg_fs_sorted) // 2],
                            reg_fs_sorted[min(len(reg_fs_sorted) - 1, max(0, len(reg_fs_sorted) // 5))],
                            reg_fs_sorted[min(len(reg_fs_sorted) - 1, (4 * len(reg_fs_sorted)) // 5)],
                            reg_fs_sorted[-1],
                        ]
                        for sp_f in span_picks:
                            if all(abs(sp_f - kept_f) >= 10 for kept_f in cand_4k_frames):
                                cand_4k_frames.append(sp_f)
                            if len(cand_4k_frames) >= 5:
                                break
                    else:
                        vrange = track.get("video_frame_range")
                        backup_fs: List[int] = []
                        if track.get("pass_frame") is not None:
                            backup_fs.append(max(0, int(track["pass_frame"])))
                        if isinstance(vrange, (list, tuple)) and len(vrange) >= 2:
                            f_lo, f_hi = max(0, int(vrange[0])), min(max(0, vid_total - 1), int(vrange[-1]))
                            if f_hi > f_lo:
                                backup_fs.extend([(f_lo + f_hi) // 2, f_hi - 8, f_lo + 8])
                        for bf in backup_fs:
                            if 0 <= bf < max(1, vid_total) and all(abs(bf - kept_f) >= 8 for kept_f in cand_4k_frames):
                                cand_4k_frames.append(bf)
                            if len(cand_4k_frames) >= 4:
                                break

                allow_num_rot = (detected_c_type == "TOP")
                for fc in cand_4k_frames:
                    cap_4k.set(cv2.CAP_PROP_POS_FRAMES, fc)
                    ok_4k, frame_4k = cap_4k.read()
                    if not ok_4k or frame_4k is None:
                        continue

                    r_fc = reg_by_frame.get(fc)
                    H_fc = np.asarray(r_fc["H_pano_to_vid"], dtype=np.float64) if (r_fc and "H_pano_to_vid" in r_fc) else None
                    cont_box_fc: Optional[Tuple[float, float, float, float]] = None
                    if r_fc and r_fc.get("det_bbox"):
                        db = r_fc["det_bbox"]
                        cont_box_fc = (float(db[0]), float(db[1]), float(db[2]), float(db[3]))
                    elif H_fc is not None and isinstance(track.get("panorama_container_bbox"), (list, tuple)):
                        pb_c = track["panorama_container_bbox"]
                        if len(pb_c) >= 4:
                            cont_box_fc = _project_box_homography((float(pb_c[0]), float(pb_c[1]), float(pb_c[2]) - float(pb_c[0]), float(pb_c[3]) - float(pb_c[1])), H_fc)

                    det_4k = detector.detect(
                        frame_4k,
                        conf_thresh=text_thresh,
                        imgsz=self.config.detection.imgsz_text_detection,
                        classes=[1, 2],
                    )
                    guard_cb = cont_box_fc
                    if guard_cb is None and reg_by_frame:
                        rf_near = min(reg_by_frame.keys(), key=lambda k: abs(k - fc))
                        db_n = reg_by_frame[rf_near].get("det_bbox")
                        if db_n and abs(fc - rf_near) <= 15:
                            pad_ext = min(150.0, 15.0 * abs(fc - rf_near))
                            guard_cb = (
                                max(0.0, float(db_n[0]) - pad_ext),
                                max(0.0, float(db_n[1]) - 15.0),
                                min(float(vid_w), float(db_n[2]) + pad_ext),
                                min(float(vid_h), float(db_n[3]) + 15.0),
                            )

                    # Strict spatial ownership (+12px max) prevents stealing adjacent container numbers
                    in_nums_4k: List[Tuple[int, int, int, int, float]] = []
                    in_isos_4k: List[Tuple[int, int, int, int, float]] = []
                    for b in det_4k:
                        bcx = (b.x1 + b.x2) / 2.0
                        bcy = (b.y1 + b.y2) / 2.0
                        if guard_cb is not None:
                            if not (guard_cb[0] - 12 <= bcx <= guard_cb[2] + 12 and guard_cb[1] - 12 <= bcy <= guard_cb[3] + 12):
                                continue
                        if b.class_id == 2:
                            in_nums_4k.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(b.confidence)))
                        elif b.class_id == 1:
                            in_isos_4k.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(b.confidence)))

                    pnb_4k: Optional[Tuple[int, int, int, int]] = None
                    if H_fc is not None and H_fc.shape == (3, 3) and nb_pano_xywh is not None:
                        pnb = _project_box_homography(nb_pano_xywh, H_fc)
                        if pnb is not None:
                            px1, py1, px2, py2 = int(round(pnb[0])), int(round(pnb[1])), int(round(pnb[2])), int(round(pnb[3]))
                            if px2 - px1 >= 10 and py2 - py1 >= 10 and px2 > 0 and py2 > 0 and px1 < vid_w and py1 < vid_h:
                                pnb_4k = (max(0, px1), max(0, py1), min(vid_w, px2), min(vid_h, py2))

                    # High-res local 4K ROI crop detection around pnb_4k so small vertical digits aren't lost in 480p downscale
                    zoom_nums_4k: List[Tuple[int, int, int, int, float]] = []
                    if pnb_4k is not None:
                        pcx = (pnb_4k[0] + pnb_4k[2]) // 2
                        pcy = (pnb_4k[1] + pnb_4k[3]) // 2
                        zx1 = max(0, pcx - 420)
                        zy1 = max(0, pcy - 420)
                        zx2 = min(vid_w, pcx + 420)
                        zy2 = min(vid_h, pcy + 420)
                        if (zx2 - zx1) >= 120 and (zy2 - zy1) >= 120:
                            z_crop = frame_4k[zy1:zy2, zx1:zx2]
                            z_dets = detector.detect(
                                z_crop,
                                conf_thresh=max(0.15, text_thresh - 0.05),
                                imgsz=self.config.detection.imgsz_text_detection,
                                classes=[1, 2],
                            )
                            for zb in z_dets:
                                gx1, gy1, gx2, gy2 = int(zx1 + zb.x1), int(zy1 + zb.y1), int(zx1 + zb.x2), int(zy1 + zb.y2)
                                gbcx, gbcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
                                if guard_cb is not None and not (guard_cb[0] - 30 <= gbcx <= guard_cb[2] + 30 and guard_cb[1] - 18 <= gbcy <= guard_cb[3] + 18):
                                    continue
                                if zb.class_id == 2:
                                    zoom_nums_4k.append((gx1, gy1, gx2, gy2, float(zb.confidence)))
                                elif zb.class_id == 1:
                                    in_isos_4k.append((gx1, gy1, gx2, gy2, float(zb.confidence)))

                    # Rank detections matching zoomed/panorama region first and penalize ISO-code overlap
                    def _num_rank(nb_t: Tuple[int, int, int, int, float]) -> Tuple[int, float]:
                        cx_n = (nb_t[0] + nb_t[2]) / 2.0
                        cy_n = (nb_t[1] + nb_t[3]) / 2.0
                        matches_pnb = 0
                        if pnb_4k is not None:
                            pcx_r = (pnb_4k[0] + pnb_4k[2]) / 2.0
                            pcy_r = (pnb_4k[1] + pnb_4k[3]) / 2.0
                            if abs(cx_n - pcx_r) < 160 and abs(cy_n - pcy_r) < 160:
                                matches_pnb = 1
                        overlaps_iso = 0
                        for iso_b in in_isos_4k:
                            ix_i = max(0, min(nb_t[2], iso_b[2]) - max(nb_t[0], iso_b[0]))
                            iy_i = max(0, min(nb_t[3], iso_b[3]) - max(nb_t[1], iso_b[1]))
                            n_area = max(1, (nb_t[2] - nb_t[0]) * (nb_t[3] - nb_t[1]))
                            if (ix_i * iy_i) / float(n_area) > 0.30:
                                overlaps_iso = 1
                                break
                        seam_margin = 1
                        if guard_cb is not None:
                            if cy_n < guard_cb[1] + 18 or cy_n > guard_cb[3] - 18:
                                seam_margin = 0
                        return (matches_pnb * 3 + (1 - overlaps_iso) * 2 + seam_margin, nb_t[4])

                    all_nums_4k = zoom_nums_4k if zoom_nums_4k else in_nums_4k
                    all_nums_4k.sort(key=_num_rank, reverse=True)
                    in_isos_4k.sort(key=lambda x: x[4], reverse=True)

                    cand_4k_boxes = list(all_nums_4k)
                    if detected_c_type == "LEFT_RIGHT" and len(all_nums_4k) >= 2:
                        s_x1, s_y1, s_x2, s_y2, s_c = all_nums_4k[0]
                        scx = (s_x1 + s_x2) / 2.0
                        collinear = [b for b in all_nums_4k if abs((b[0] + b[2]) / 2.0 - scx) < 90]
                        if len(collinear) >= 2:
                            merged = (
                                min(b[0] for b in collinear),
                                min(b[1] for b in collinear),
                                max(b[2] for b in collinear),
                                max(b[3] for b in collinear),
                                s_c,
                            )
                            cand_4k_boxes.insert(0, merged)

                    if pnb_4k is not None:
                        pnb_cand = (pnb_4k[0], pnb_4k[1], pnb_4k[2], pnb_4k[3], 0.90)
                        has_clean_4k = bool(all_nums_4k) and _num_rank(all_nums_4k[0])[0] >= 5
                        if has_clean_4k:
                            cand_4k_boxes.append(pnb_cand)
                        elif 40 <= pnb_4k[0] and pnb_4k[2] <= vid_w - 40:
                            cand_4k_boxes.insert(0, pnb_cand)
                        elif not any(abs((b[0] + b[2]) / 2.0 - (pnb_4k[0] + pnb_4k[2]) / 2.0) < 90 and abs((b[1] + b[3]) / 2.0 - (pnb_4k[1] + pnb_4k[3]) / 2.0) < 90 for b in cand_4k_boxes):
                            cand_4k_boxes.insert(0, pnb_cand)

                    for (nx1, ny1, nx2, ny2, n_conf) in cand_4k_boxes[:4]:
                        bw = nx2 - nx1
                        bh = ny2 - ny1
                        if bw < 8 or bh < 8:
                            continue
                        pad_x = max(8, int(bw * 0.16))
                        pad_y = max(8, int(bh * 0.18))
                        cx1, cy1 = max(0, nx1 - pad_x), max(0, ny1 - pad_y)
                        cx2, cy2 = min(vid_w, nx2 + pad_x), min(vid_h, ny2 + pad_y)
                        crop_4k = frame_4k[cy1:cy2, cx1:cx2].copy()
                        if crop_4k is None or crop_4k.size == 0:
                            continue

                        prev_len = len(ocr_state["clean_text"])
                        hit_valid = _run_ocr_num_crop(
                            crop_4k, route_order, detected_c_type, effective_cam_pos, f"4K-F{fc}", allow_num_rot, ocr_state
                        )
                        if hit_valid or len(ocr_state["clean_text"]) > prev_len or best_4k_num_box is None:
                            best_4k_frame = fc
                            best_4k_num_box = (nx1, ny1, nx2, ny2)
                            best_4k_cont_box = cont_box_fc or guard_cb
                            if H_fc is not None and H_fc.shape == (3, 3):
                                try:
                                    H_inv = np.linalg.inv(H_fc)
                                    p_back = _project_box_homography((float(nx1), float(ny1), float(bw), float(bh)), H_inv)
                                    if p_back is not None and 0 <= p_back[0] < cc_w and 0 <= p_back[1] < cc_h:
                                        best_num_box = (
                                            max(0, int(round(p_back[0]))),
                                            max(0, int(round(p_back[1]))),
                                            min(cc_w, int(round(p_back[2]))),
                                            min(cc_h, int(round(p_back[3]))),
                                        )
                                except np.linalg.LinAlgError:
                                    pass
                            iso_4k_box = in_isos_4k[0][:4] if in_isos_4k else None
                            if iso_4k_box is None and H_fc is not None and H_fc.shape == (3, 3) and iso_pano_xywh is not None:
                                pi4k = _project_box_homography(iso_pano_xywh, H_fc)
                                if pi4k is not None and pi4k[2] - pi4k[0] >= 8 and pi4k[3] - pi4k[1] >= 8:
                                    iso_4k_box = (max(0, int(round(pi4k[0]))), max(0, int(round(pi4k[1]))), min(vid_w, int(round(pi4k[2]))), min(vid_h, int(round(pi4k[3]))))
                            if iso_4k_box is not None:
                                ix1, iy1, ix2, iy2 = iso_4k_box
                                iw, ih = ix2 - ix1, iy2 - iy1
                                ipx, ipy = max(4, int(iw * 0.12)), max(2, int(ih * 0.08))
                                best_4k_iso_crop = frame_4k[max(0, iy1 - ipy):min(vid_h, iy2 + ipy), max(0, ix1 - ipx):min(vid_w, ix2 + ipx)].copy()
                                if H_fc is not None and H_fc.shape == (3, 3):
                                    try:
                                        pi_back = _project_box_homography((float(ix1), float(iy1), float(iw), float(ih)), np.linalg.inv(H_fc))
                                        if pi_back is not None and 0 <= pi_back[0] < cc_w and 0 <= pi_back[1] < cc_h:
                                            best_4k_iso_box_pano = (max(0, int(round(pi_back[0]))), max(0, int(round(pi_back[1]))), min(cc_w, int(round(pi_back[2]))), min(cc_h, int(round(pi_back[3]))))
                                    except np.linalg.LinAlgError:
                                        pass
                        if hit_valid:
                            break
                    if ocr_state["is_valid"]:
                        break

                if not ocr_state["is_valid"] and candidate_num_boxes:
                    for (nx1, ny1, nx2, ny2, _) in candidate_num_boxes[:3]:
                        bw = nx2 - nx1
                        bh = ny2 - ny1
                        if bw < 8 or bh < 8:
                            continue
                        pad_x = max(4, int(bw * 0.15))
                        pad_y = max(6, int(bh * 0.18))
                        crop_c = raw_cv_img[max(0, ny1 - pad_y):min(cc_h, ny2 + pad_y), max(0, nx1 - pad_x):min(cc_w, nx2 + pad_x)].copy()
                        if crop_c is None or crop_c.size == 0:
                            continue
                        prev_len = len(ocr_state["clean_text"])
                        hit_valid = _run_ocr_num_crop(
                            crop_c, route_order, detected_c_type, effective_cam_pos, "PANO", allow_num_rot, ocr_state
                        )
                        if hit_valid or len(ocr_state["clean_text"]) > prev_len or best_num_box is None:
                            best_num_box = (nx1, ny1, nx2, ny2)
                        if hit_valid:
                            break

                clean_text = ocr_state["clean_text"]
                raw_text = ocr_state["raw_text"]
                is_valid = ocr_state["is_valid"]
                best_conf = ocr_state["best_conf"]
                if ocr_state["best_img"] is not None:
                    t_num_p = temp_crop_dir / f"job{job_id}_reg{route_order:03d}_num.jpg"
                    imwrite_unicode(t_num_p, ocr_state["best_img"])
                    num_crop_path = t_num_p

                if best_4k_frame is not None and best_4k_num_box is not None:
                    track["direct_4k_anchor"] = {
                        "frame": int(best_4k_frame),
                        "num_box": best_4k_num_box,
                        "cont_box": best_4k_cont_box,
                    }

                # 3. Proses OCR untuk iso_code (4K terlebih dahulu, lalu fallback panorama)
                iso_text = ""
                iso_crop_path: Optional[Path] = None
                best_iso_box: Optional[Tuple[int, int, int, int]] = (
                    best_4k_iso_box_pano
                    or ((candidate_iso_boxes[0][0], candidate_iso_boxes[0][1], candidate_iso_boxes[0][2], candidate_iso_boxes[0][3]) if candidate_iso_boxes else None)
                )
                best_iso_conf = 0.0

                crop_iso_eval = None
                if best_4k_iso_crop is not None and getattr(best_4k_iso_crop, "size", 0) > 0:
                    crop_iso_eval = best_4k_iso_crop
                elif candidate_iso_boxes:
                    ix1, iy1, ix2, iy2, _ = candidate_iso_boxes[0]
                    iw, ih = ix2 - ix1, iy2 - iy1
                    pad_x, pad_y = max(3, int(iw * 0.15)), max(4, int(ih * 0.15))
                    crop_iso_eval = raw_cv_img[max(0, iy1 - pad_y):min(cc_h, iy2 + pad_y), max(0, ix1 - pad_x):min(cc_w, ix2 + pad_x)].copy()

                if crop_iso_eval is not None and crop_iso_eval.size > 0:
                    t_iso_p = temp_crop_dir / f"job{job_id}_reg{route_order:03d}_iso.jpg"
                    if detected_c_type == "TOP" and crop_iso_eval.shape[0] > crop_iso_eval.shape[1] * 1.1:
                        crop_iso_eval = cv2.rotate(
                            crop_iso_eval,
                            cv2.ROTATE_90_COUNTERCLOCKWISE if preferred_top_rot[0] == "CCW" else cv2.ROTATE_90_CLOCKWISE,
                        )
                    if min(crop_iso_eval.shape[:2]) < 32:
                        crop_iso_eval = cv2.resize(crop_iso_eval, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    imwrite_unicode(t_iso_p, crop_iso_eval)
                    try:
                        it, i_api_conf = self.ocr_client.recognize_image_file(
                            t_iso_p,
                            detection_type="iso_code",
                            camera_position=effective_cam_pos,
                            req_id=job_id,
                            log_callback=log_cb,
                            context_info=f"Kontainer R{route_order:03d} ISO ({detected_c_type})",
                        )
                        iso_text = it.strip().upper()
                        iso_crop_path = t_iso_p
                        best_iso_conf = i_api_conf
                    except Exception:
                        pass

                # 4. Anotasi visual utuh beresolusi tinggi (Standar identik dengan FRONT_REAR)
                annotated_crop = raw_cv_img.copy()
                # Border hijau kontainer
                cv2.rectangle(annotated_crop, (2, 2), (cc_w - 3, cc_h - 3), (0, 255, 127), 3)
                label_header = f"Kontainer R{route_order:03d} ({detected_c_type})"
                cv2.rectangle(annotated_crop, (2, 2), (min(cc_w - 3, 2 + len(label_header) * 13), 32), (0, 100, 40), -1)
                cv2.putText(
                    annotated_crop,
                    label_header,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                # Bounding box nomor kontainer warna cyan tepat pada teks
                if best_num_box is not None:
                    bx1, by1, bx2, by2 = best_num_box
                    cv2.rectangle(annotated_crop, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                    disp_text = clean_text if clean_text else (raw_text if raw_text else "NO_OCR")
                    cv2.rectangle(annotated_crop, (bx1, max(0, by1 - 24)), (min(cc_w, bx1 + len(disp_text) * 12), by1), (0, 0, 0), -1)
                    cv2.putText(
                        annotated_crop,
                        disp_text,
                        (bx1 + 2, max(16, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                # Bounding box ISO code warna oranye jika ada
                if best_iso_box is not None:
                    ix1, iy1, ix2, iy2 = best_iso_box
                    cv2.rectangle(annotated_crop, (ix1, iy1), (ix2, iy2), (0, 165, 255), 2)
                    disp_iso = iso_text if iso_text else "ISO"
                    cv2.rectangle(annotated_crop, (ix1, max(0, iy1 - 22)), (min(cc_w, ix1 + len(disp_iso) * 11), iy1), (0, 0, 0), -1)
                    cv2.putText(
                        annotated_crop,
                        disp_iso,
                        (ix1 + 2, max(16, iy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 165, 255),
                        2,
                    )

                track["clean_text"] = clean_text
                track["iso_text"] = iso_text
                track["is_valid"] = is_valid
                if best_num_box is not None:
                    track["best_num_box"] = best_num_box
                    track.setdefault("detections", {})["container_number"] = {
                        "bbox": {
                            "x": int(best_num_box[0]),
                            "y": int(best_num_box[1]),
                            "width": int(best_num_box[2] - best_num_box[0]),
                            "height": int(best_num_box[3] - best_num_box[1]),
                        },
                        "confidence": float(best_conf),
                    }

                # 5. Simpan gambar terstandarisasi ke folder images/ (Identik dengan FRONT_REAR)
                clean_tag = clean_text if (clean_text and clean_text != "-") else f"track{container_id:02d}"
                iso_tag = f"_{iso_text}" if (iso_text and iso_text != "-") else ""
                base_name = f"{video_stem}_{timestamp_str}_R{route_order:03d}_{clean_tag}{iso_tag}"

                ann_crop_path = images_dir / f"{base_name}_annotated.jpg"
                raw_crop_path = images_dir / f"{base_name}_raw.jpg"
                imwrite_unicode(ann_crop_path, annotated_crop)
                imwrite_unicode(raw_crop_path, raw_cv_img)
                track["image_path"] = str(raw_crop_path)
                track["annotated_image_path"] = str(ann_crop_path)

                # 6. Simpan kontainer dan deteksi ke SQLite Database
                c_db_id = self.db.add_container(
                    job_id=job_id,
                    route_order=route_order,
                    container_id=container_id,
                    status="stable",
                    image_path=str(raw_crop_path),
                    annotated_image_path=str(ann_crop_path),
                    pos_x=int(pos.get("x", 0)),
                    pos_y=int(pos.get("y", 0)),
                    pos_w=int(pos.get("width", cc_w)),
                    pos_h=int(pos.get("height", cc_h)),
                    container_type=detected_c_type,
                )

                self.db.add_detection(
                    container_id=c_db_id,
                    detection_type="container_number",
                    bbox_x=int(best_num_box[0]) if best_num_box else 0,
                    bbox_y=int(best_num_box[1]) if best_num_box else 0,
                    bbox_w=int(best_num_box[2] - best_num_box[0]) if best_num_box else 0,
                    bbox_h=int(best_num_box[3] - best_num_box[1]) if best_num_box else 0,
                    confidence=float(best_conf),
                    crop_image_path=str(num_crop_path) if num_crop_path else str(raw_crop_path),
                    ocr_raw_text=raw_text or "NO_TEXT_DETECTED",
                    ocr_clean_text=clean_text or "-",
                    is_valid_iso=is_valid,
                )

                self.db.add_detection(
                    container_id=c_db_id,
                    detection_type="iso_code",
                    bbox_x=int(best_iso_box[0]) if best_iso_box else 0,
                    bbox_y=int(best_iso_box[1]) if best_iso_box else 0,
                    bbox_w=int(best_iso_box[2] - best_iso_box[0]) if best_iso_box else 0,
                    bbox_h=int(best_iso_box[3] - best_iso_box[1]) if best_iso_box else 0,
                    confidence=float(best_iso_conf),
                    crop_image_path=str(iso_crop_path) if iso_crop_path else None,
                    ocr_raw_text=iso_text or "",
                    ocr_clean_text=iso_text.strip().upper() if iso_text else "-",
                    is_valid_iso=False,
                )

                # 7. Update progress dan event ke UI
                v_badge = "VALID ISO 6346" if is_valid else ("CHECK DIGIT TIDAK SESUAI" if clean_text and clean_text != "-" else "NO TEXT")
                log_num_msg = f"[OCR PARALEL] R{route_order:03d} ({detected_c_type}): Nomor '{clean_text or '-'}' ({v_badge})"
                if iso_text:
                    log_num_msg += f" | ISO: '{iso_text}'"

                self.db.update_job_status(job_id, "PROCESSING", processed_regions=idx)
                pct = 0.85 + (idx / max(1, total_regions)) * 0.14
                self._emit_event(
                    OrchestratorEvent(
                        type="PROGRESS",
                        job_id=job_id,
                        data={
                            "stage": "OCR",
                            "message": f"Tahap 4/4 (OCR Cloud API & Panorama): Kontainer {idx}/{total_regions}...",
                            "progress": pct,
                            "log": log_num_msg,
                        },
                    )
                )
                self._emit_event(
                    OrchestratorEvent(
                        type="CONTAINER_PROCESSED",
                        job_id=job_id,
                        data={"route_order": route_order, "processed": idx, "total": total_regions},
                    )
                )

            if cap_4k.isOpened():
                cap_4k.release()

            # Rapihkan folder output: sembunyikan file intermediate stitching CLI
            raw_mode_dir = job_output_dir / mode
            if raw_mode_dir.is_dir():
                stitching_hidden_dir = job_output_dir / ".stitching"
                if stitching_hidden_dir.exists():
                    shutil.rmtree(stitching_hidden_dir, ignore_errors=True)
                try:
                    raw_mode_dir.rename(stitching_hidden_dir)
                except Exception:
                    pass
            old_crops_4k = job_output_dir / "crops_4k"
            if old_crops_4k.is_dir():
                shutil.rmtree(old_crops_4k, ignore_errors=True)

            # Render video hasil anotasi stitching
            out_vid_path = job_output_dir / f"{video_stem}_annotated.mp4"
            try:
                self._render_annotated_video_stitching(
                    video_path=video_path,
                    output_video_path=out_vid_path,
                    tracks_data=tracks_data_holder,
                    job_id=job_id,
                    mode=mode,
                )
            except Exception as e:
                self._emit_event(
                    OrchestratorEvent(
                        type="LOG",
                        job_id=job_id,
                        data={"log": f"[WARN] Gagal merender video anotasi: {e}"},
                    )
                )

        finally:
            if cap_4k.isOpened():
                cap_4k.release()
            audit_logger.finalize()

    def _process_single_crop(
        self,
        c_db_id: int,
        det_type: str,
        det_data: Dict[str, Any],
        raw_cv_img: Any,
        temp_crop_dir: Path,
        file_stem: str,
        camera_position: str = "rear",
        direct_crop: Optional[Any] = None,
        audit_logger: Optional[ApiAuditLogger] = None,
        context_info: Optional[str] = None,
    ) -> Tuple[str, bool]:
        bbox = det_data.get("bbox", {})
        conf = float(det_data.get("confidence", 0.0))

        if direct_crop is not None and getattr(direct_crop, "size", 0) > 0:
            cropped = direct_crop
        elif raw_cv_img is not None:
            # Fallback crop dari citra panorama dengan margin padding aman
            img_h, img_w = raw_cv_img.shape[:2]
            bx = int(bbox.get("x", 0))
            by = int(bbox.get("y", 0))
            bw = int(bbox.get("width", 0))
            bh = int(bbox.get("height", 0))
            if det_type == "container_number":
                pad_top = int(bh * 0.25)
                pad_bot = int(bh * 0.08)
                pad_side = int(bw * 0.15)
                px1 = max(0, bx - pad_side)
                py1 = max(0, by - pad_top)
                px2 = min(img_w, bx + bw + pad_side)
                py2 = min(img_h, by + bh + pad_bot)
            else:
                pad_x = max(3, int(bw * 0.15))
                pad_y = max(4, int(bh * 0.15))
                px1 = max(0, bx - pad_x)
                py1 = max(0, by - pad_y)
                px2 = min(img_w, bx + bw + pad_x)
                py2 = min(img_h, by + bh + pad_y)
            cropped = raw_cv_img[py1:py2, px1:px2]
        else:
            return "", False

        crop_path = temp_crop_dir / f"{file_stem}.jpg"
        imwrite_unicode(crop_path, cropped)

        raw_ocr_text = ""
        clean_text = ""
        is_valid = False

        # AI Enhancement: Orientasi teks pada nomor kontainer
        # Hanya kontainer atap (TOP) yang teks nomornya tercetak miring 90 derajat (bh > bw * 1.2).
        # Untuk LEFT_RIGHT & FRONT_REAR teks sudah tegak lurus, JANGAN diputar horizontal.
        is_vertical_text = (
            det_type == "container_number"
            and camera_position == "top"
            and cropped is not None
            and hasattr(cropped, "shape")
            and cropped.shape[0] > cropped.shape[1] * 1.2
        )
        effective_cam_pos = "rear" if camera_position in ("top", "front_rear") else camera_position

        try:
            log_cb = audit_logger.on_api_event if audit_logger else None

            if is_vertical_text:
                # 1. Coba rotasi searah jarum jam (CW)
                crop_cw = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
                crop_path_cw = temp_crop_dir / f"{file_stem}_cw.jpg"
                imwrite_unicode(crop_path_cw, crop_cw)

                text_cw, _ = self.ocr_client.recognize_image_file(
                    crop_path_cw,
                    detection_type=det_type,
                    camera_position=effective_cam_pos,
                    req_id=c_db_id,
                    log_callback=log_cb,
                    context_info=f"{context_info or ''} (Rot CW)",
                )
                iso_cw = validate_iso6346(text_cw)

                if iso_cw.is_valid:
                    raw_ocr_text = text_cw
                    clean_text = iso_cw.clean_number
                    is_valid = True
                    imwrite_unicode(crop_path, crop_cw)
                else:
                    # 2. Coba rotasi berlawanan jarum jam (CCW)
                    crop_ccw = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    crop_path_ccw = temp_crop_dir / f"{file_stem}_ccw.jpg"
                    imwrite_unicode(crop_path_ccw, crop_ccw)

                    text_ccw, _ = self.ocr_client.recognize_image_file(
                        crop_path_ccw,
                        detection_type=det_type,
                        camera_position=effective_cam_pos,
                        req_id=c_db_id,
                        log_callback=log_cb,
                        context_info=f"{context_info or ''} (Rot CCW)",
                    )
                    iso_ccw = validate_iso6346(text_ccw)

                    if iso_ccw.is_valid or len(iso_ccw.clean_number) > len(iso_cw.clean_number):
                        raw_ocr_text = text_ccw
                        clean_text = iso_ccw.clean_number
                        is_valid = iso_ccw.is_valid
                        imwrite_unicode(crop_path, crop_ccw)
                    else:
                        raw_ocr_text = text_cw
                        clean_text = iso_cw.clean_number
                        is_valid = iso_cw.is_valid
                        imwrite_unicode(crop_path, crop_cw)

                # Bersihkan temporary rotation crops
                try:
                    crop_path_cw.unlink(missing_ok=True)
                    crop_path_ccw.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                text, _ = self.ocr_client.recognize_image_file(
                    crop_path,
                    detection_type=det_type,
                    camera_position=effective_cam_pos,
                    req_id=c_db_id,
                    log_callback=log_cb,
                    context_info=context_info,
                )
                raw_ocr_text = text
                if det_type == "container_number":
                    iso_res = validate_iso6346(raw_ocr_text)
                    clean_text = iso_res.clean_number
                    is_valid = iso_res.is_valid
                else:
                    clean_text = raw_ocr_text.strip().upper()
        except OcrClientError as e:
            raw_ocr_text = f"[OCR ERROR: {e}]"
            clean_text = ""

        self.db.add_detection(
            container_id=c_db_id,
            detection_type=det_type,
            bbox_x=int(bbox.get("x", 0)),
            bbox_y=int(bbox.get("y", 0)),
            bbox_w=int(bbox.get("width", 0)),
            bbox_h=int(bbox.get("height", 0)),
            confidence=conf,
            crop_image_path=str(crop_path),
            ocr_raw_text=raw_ocr_text,
            ocr_clean_text=clean_text,
            is_valid_iso=is_valid,
        )
        return clean_text, is_valid

    def _process_front_rear_job(
        self,
        job_id: int,
        video_path: str,
        camera_position: str = "rear",
        job_output_dir: Optional[Path] = None,
    ) -> None:
        """
        Processes FRONT_REAR mode (and direct video modes) with a 2-stage hierarchical pipeline:
        Stage 1: Container Detection & Tracking
          - Scans video frames and detects containers (classes=[0, 3, 4]).
          - Tracks container bounding boxes across frames to group observations of the same container.
          - Associates container_number and iso_code detected strictly INSIDE each container.
        Stage 2: Best Observation Extraction, Cloud OCR & Visual Annotation
          - Selects the sharpest, highest-confidence frame observation for each container track.
          - Crops the container with a 6% padding margin.
          - Extracts high-resolution ROI crops of container_number and iso_code.
          - Submits crops to Halotec OCR Cloud API with matching camera_position.
          - Draws clear visual annotations directly on the container crop (green container border, cyan number ROI, orange ISO ROI).
          - Persists annotated_image_path to DB and updates the UI table and inspection panel in real-time.
        """
        video = Path(video_path).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Video file not found: {video}")

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")

        detector = self._get_detector()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(video_path).stem
        if job_output_dir is None:
            job_output_dir = self.config.paths.resolved_output_dir() / f"{timestamp_str}_{video_stem}"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = job_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        temp_crop_dir = job_output_dir / ".temp_crops"
        temp_crop_dir.mkdir(parents=True, exist_ok=True)
        job_logs_dir = job_output_dir / "logs"
        job_logs_dir.mkdir(parents=True, exist_ok=True)
        api_log_json_path = job_logs_dir / "api_requests.json"
        activity_log_path = job_logs_dir / "job_activity.log"
        if self._active_log_path is None:
            self._active_log_path = activity_log_path
            self._active_job_start = datetime.now()
            self._log_activity(activity_log_path, f"Job #{job_id} (FRONT_REAR, pos={camera_position}) dimulai. Video: {video_path}")
        audit_logger = ApiAuditLogger(
            txt_path=activity_log_path,
            json_path=api_log_json_path,
            evt_queue=self.evt_queue,
            job_id=job_id,
            emit_fn=self._emit_event,
            write_txt=False,
        )
        self._active_audit_logger = audit_logger

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_seconds = max(0.1, min(0.5, float(getattr(self.config.detection, "frame_sample_interval_seconds", 0.5))))
        sample_interval = max(1, int(fps * sample_seconds))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Hitung gerak kamera kumulatif drone C[i] di awal (LK optical flow + affine RANSAC)
        # Digunakan untuk Motion-Compensated IoU Tracking di Phase 1 dan Video Renderer di Phase 2
        self._emit_event(
            OrchestratorEvent(
                type="PROGRESS",
                job_id=job_id,
                data={
                    "stage": "FRONT_REAR",
                    "message": "Menghitung gerak kamera drone kumulatif...",
                    "progress": 0.05,
                    "log": f"Video: {video.name} ({total_frames} frame). Menghitung gerak kamera LK optical flow...",
                },
            )
        )
        C_motion = estimate_cumulative_motion(
            str(video_path),
            cancel=self._cancel_requested.is_set,
            progress_callback=lambda pct, msg: self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={"stage": "FRONT_REAR", "message": msg, "progress": 0.05 + 0.10 * pct, "log": f"[MOTION] {msg}"},
                )
            ),
        )

        self._emit_event(
            OrchestratorEvent(
                type="PROGRESS",
                job_id=job_id,
                data={
                    "stage": "FRONT_REAR",
                    "message": f"Tahap 1/2: Memulai tracking kontainer dari video (interval {sample_seconds:.1f}s)...",
                    "progress": 0.15,
                    "log": f"Tracking terkompensasi gerak aktif (interval sampling={sample_seconds:.1f}s / {sample_interval} frame)",
                },
            )
        )

        # -------------------------------------------------------------
        # PHASE 1: CONTAINER-CENTRIC TRACKING & GOLDEN FRAME SELECTION
        # (Zero frame-drift, motion-compensated IoU gating against neighbor jumping)
        # -------------------------------------------------------------
        class _ContainerTrack:
            def __init__(self, t_id: int, cb: Any, frame_num: int):
                self.track_id = t_id
                self.last_cb = cb
                self.last_frame_idx = frame_num
                self.disappeared = 0
                self.obs_count = 1
                self.first_frame_idx = frame_num
                self.best_score = -1.0
                self.best_frame_idx = frame_num
                self.best_cb = cb
                self.best_nb: Optional[Any] = None
                self.best_ib: Optional[Any] = None
                self.best_container_crop: Optional[np.ndarray] = None
                self.best_num_crop: Optional[np.ndarray] = None
                self.best_iso_crop: Optional[np.ndarray] = None
                self.is_utuh = False
                self.clean_text: str = ""
                self.iso_text: str = ""
                self.is_valid: bool = False
                self.history: Dict[int, Any] = {}
                self.num_history: Dict[int, Any] = {}

        tracks: List[_ContainerTrack] = []
        next_track_id = 1

        def _populate_fr_text_crops(
            track_obj: _ContainerTrack,
            frame_img: np.ndarray,
            cb_box: Any,
            nums_in: List[Any],
            isos_in: List[Any],
            f_idx: int,
            im_w: int,
            im_h: int,
        ) -> None:
            if nums_in:
                right_nums = [nb for nb in nums_in if (nb.x1 + nb.x2) / 2.0 >= cb_box.x1 + 0.35 * cb_box.width]
                cand_nums = right_nums if right_nums else nums_in
                seed_nb = max(cand_nums, key=lambda n: n.confidence)
                seed_cx = (seed_nb.x1 + seed_nb.x2) / 2.0
                collinear = [nb for nb in cand_nums if abs((nb.x1 + nb.x2) / 2.0 - seed_cx) < 80]
                min_nx1 = min(nb.x1 for nb in collinear)
                min_ny1 = min(nb.y1 for nb in collinear)
                max_nx2 = max(nb.x2 for nb in collinear)
                max_ny2 = max(nb.y2 for nb in collinear)
                track_obj.num_history[f_idx] = (min_nx1, min_ny1, max_nx2, max_ny2)
                nh = max_ny2 - min_ny1
                nw = max_nx2 - min_nx1
                # Tighten top/bottom padding on horizontal door text so adjacent 45G1 line doesn't bleed in
                if nw > nh * 1.4:
                    py = max(4, int(nh * 0.08))
                    px = max(14, int(nw * 0.16))
                else:
                    py = max(15, int(nh * 0.18))
                    px = max(16, int(nw * 0.22))
                ny1 = max(0, min_ny1 - py)
                ny2 = min(im_h, max_ny2 + py)
                nx1 = max(0, min_nx1 - px)
                nx2 = min(im_w, max_nx2 + px)
                track_obj.best_num_crop = frame_img[ny1:ny2, nx1:nx2].copy()
                track_obj.best_nb = DetectionBox(
                    seed_nb.class_id,
                    seed_nb.class_name,
                    seed_nb.confidence,
                    min_nx1,
                    min_ny1,
                    max_nx2,
                    max_ny2,
                )

            if isos_in:
                best_iso = max(isos_in, key=lambda i: i.confidence)
                iw = max(1, int(best_iso.width))
                ih = max(1, int(best_iso.height))
                ix_pad = max(5, int(iw * 0.12))
                iy_pad = 2
                iy1 = max(0, int(best_iso.y1) - iy_pad)
                iy2 = min(im_h, int(best_iso.y2) + iy_pad)
                ix1 = max(0, int(best_iso.x1) - ix_pad)
                ix2 = min(im_w, int(best_iso.x2) + ix_pad)
                track_obj.best_iso_crop = frame_img[iy1:iy2, ix1:ix2].copy()
                track_obj.best_ib = best_iso

        frame_idx = 0
        try:
            while True:
                if self._cancel_requested.is_set():
                    self.db.update_job_status(job_id, "INTERRUPTED", error_message="Dibatalkan oleh pengguna")
                    raise RuntimeError("Proses dibatalkan oleh pengguna")

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % sample_interval != 0:
                    continue

                img_h, img_w = frame.shape[:2]

                # Deteksi multi-class ONNX CPU (480x480 native model input)
                boxes = detector.detect(
                    frame,
                    conf_thresh=0.20,
                    imgsz=480,
                    classes=[0, 1, 2, 3, 4],
                )

                # Filter kotak badan kontainer (class 0: left_right, 3: rear_front, 4: top) + cross-class NMS
                raw_c_boxes = sorted(
                    [b for b in boxes if b.class_id in (0, 3, 4) and b.width >= 200 and b.height >= 250],
                    key=lambda b: b.confidence,
                    reverse=True,
                )
                c_boxes = []
                for cand_cb in raw_c_boxes:
                    cb_tup = (float(cand_cb.x1), float(cand_cb.y1), float(cand_cb.x2), float(cand_cb.y2))
                    if all(
                        box_iou(cb_tup, (float(kept.x1), float(kept.y1), float(kept.x2), float(kept.y2))) < 0.50
                        for kept in c_boxes
                    ):
                        c_boxes.append(cand_cb)
                num_boxes = [b for b in boxes if b.class_id == 2]
                iso_boxes = [b for b in boxes if b.class_id == 1]

                matched_tracks = set()
                matched_cbs = set()

                # Global best-IoU 1-to-1 track assignment (prevents exiting tracks from hijacking incoming containers)
                cand_pairs: List[Tuple[float, int, int]] = []
                for ti, t in enumerate(tracks):
                    if t.disappeared > 4:
                        continue
                    pred_box = map_box(
                        (float(t.last_cb.x1), float(t.last_cb.y1), float(t.last_cb.x2), float(t.last_cb.y2)),
                        C_motion,
                        t.last_frame_idx - 1,
                        frame_idx - 1,
                    )
                    for cbi, cb in enumerate(c_boxes):
                        cb_box = (float(cb.x1), float(cb.y1), float(cb.x2), float(cb.y2))
                        iou_val = box_iou(pred_box, cb_box)
                        width_ratio = float(cb.width) / max(1.0, float(t.last_cb.width))
                        cy_diff = abs((cb.y1 + cb.y2) / 2.0 - (t.last_cb.y1 + t.last_cb.y2) / 2.0)
                        max_allowed_dy = max(180.0, float(t.last_cb.height) * 0.45)
                        # Guard: a track already clipped exiting on left edge cannot expand onto adjacent container
                        if t.last_cb.x1 <= 20 and cb.x1 > t.last_cb.x1 + 25 and width_ratio > 1.25:
                            continue
                        if iou_val >= 0.25 and 0.65 <= width_ratio <= 1.45 and cy_diff <= max_allowed_dy:
                            cand_pairs.append((iou_val, ti, cbi))

                cand_pairs.sort(key=lambda x: x[0], reverse=True)
                track_match_map: Dict[int, int] = {}
                for _, ti, cbi in cand_pairs:
                    t = tracks[ti]
                    if t.track_id in matched_tracks or cbi in matched_cbs:
                        continue
                    track_match_map[ti] = cbi
                    matched_tracks.add(t.track_id)
                    matched_cbs.add(cbi)

                for ti, t in enumerate(tracks):
                    if t.disappeared > 4:
                        continue
                    best_cbi = track_match_map.get(ti, -1)
                    if best_cbi >= 0:
                        cb = c_boxes[best_cbi]
                        t.last_cb = cb
                        t.last_frame_idx = frame_idx
                        t.disappeared = 0
                        t.obs_count += 1
                        t.history[frame_idx] = cb

                        # Cari nomor kontainer dan ISO code di dalam kontainer ini
                        in_nums = [
                            nb for nb in num_boxes
                            if cb.x1 - 30 <= nb.x1 and cb.x2 + 30 >= nb.x2 and cb.y1 - 30 <= nb.y1 and cb.y2 + 30 >= nb.y2
                        ]
                        in_isos = [
                            ib for ib in iso_boxes
                            if cb.x1 - 30 <= ib.x1 and cb.x2 + 30 >= ib.x2 and cb.y1 - 30 <= ib.y1 and cb.y2 + 30 >= ib.y2
                        ]

                        # Simpan nomor kontainer ke num_history untuk SETIAP frame yang mendeteksi nomor
                        cand_nums = []
                        if in_nums:
                            right_nums = [nb for nb in in_nums if (nb.x1 + nb.x2) / 2.0 >= cb.x1 + 0.35 * cb.width]
                            cand_nums = right_nums if right_nums else in_nums
                            seed_nb = max(cand_nums, key=lambda n: n.confidence)
                            seed_cx = (seed_nb.x1 + seed_nb.x2) / 2.0
                            collinear = [nb for nb in cand_nums if abs((nb.x1 + nb.x2) / 2.0 - seed_cx) < 80]
                            min_nx1 = min(nb.x1 for nb in collinear)
                            min_ny1 = min(nb.y1 for nb in collinear)
                            max_nx2 = max(nb.x2 for nb in collinear)
                            max_ny2 = max(nb.y2 for nb in collinear)
                            t.num_history[frame_idx] = (min_nx1, min_ny1, max_nx2, max_ny2)

                        # Penilaian Kualitas Golden Frame:
                        # 1. Tolak jika tepi kontainer terpotong layar (edge rejection)
                        touches_edge = (cb.x1 <= 20 or cb.x2 >= img_w - 20 or cb.y1 <= 20 or cb.y2 >= img_h - 20)
                        # 2. Sentralitas posisi (centrality)
                        center_x = (cb.x1 + cb.x2) / 2.0
                        dist_to_center = abs(center_x - (img_w / 2.0))
                        centrality = max(0.0, 1.0 - (dist_to_center / (img_w / 2.0)))

                        has_num = len(in_nums) > 0
                        score = (
                            (25.0 if not touches_edge else 0.0) +
                            (15.0 if has_num else 0.0) +
                            centrality * 6.0 +
                            float(cb.confidence) * 2.0
                        )

                        if score > t.best_score or t.best_container_crop is None:
                            t.best_score = score
                            t.best_frame_idx = frame_idx
                            t.best_cb = cb
                            t.is_utuh = not touches_edge

                            # 1. Crop kontainer utuh dari frame asli dengan margin 4%
                            mw = max(15, int(cb.width * 0.04))
                            mh = max(15, int(cb.height * 0.04))
                            crop_x1 = max(0, cb.x1 - mw)
                            crop_y1 = max(0, cb.y1 - mh)
                            crop_x2 = min(img_w, cb.x2 + mw)
                            crop_y2 = min(img_h, cb.y2 + mh)
                            t.best_container_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                            # 2 & 3. Crop nomor kontainer & ISO code presisi tinggi
                            _populate_fr_text_crops(t, frame, cb, in_nums, in_isos, frame_idx, img_w, img_h)
                    else:
                        t.disappeared += 1

                # Inisialisasi track baru untuk kontainer yang belum terasosiasi
                for cbi, cb in enumerate(c_boxes):
                    if cbi not in matched_cbs:
                        t = _ContainerTrack(next_track_id, cb, frame_idx)
                        t.history[frame_idx] = cb
                        next_track_id += 1

                        in_nums = [
                            nb for nb in num_boxes
                            if cb.x1 - 30 <= nb.x1 and cb.x2 + 30 >= nb.x2 and cb.y1 - 30 <= nb.y1 and cb.y2 + 30 >= nb.y2
                        ]
                        in_isos = [
                            ib for ib in iso_boxes
                            if cb.x1 - 30 <= ib.x1 and cb.x2 + 30 >= ib.x2 and cb.y1 - 30 <= ib.y1 and cb.y2 + 30 >= ib.y2
                        ]

                        touches_edge = (cb.x1 <= 20 or cb.x2 >= img_w - 20 or cb.y1 <= 20 or cb.y2 >= img_h - 20)
                        center_x = (cb.x1 + cb.x2) / 2.0
                        dist_to_center = abs(center_x - (img_w / 2.0))
                        centrality = max(0.0, 1.0 - (dist_to_center / (img_w / 2.0)))

                        has_num = len(in_nums) > 0
                        score = (
                            (25.0 if not touches_edge else 0.0) +
                            (15.0 if has_num else 0.0) +
                            centrality * 6.0 +
                            float(cb.confidence) * 2.0
                        )
                        t.best_score = score
                        t.best_cb = cb
                        t.is_utuh = not touches_edge

                        mw = max(15, int(cb.width * 0.04))
                        mh = max(15, int(cb.height * 0.04))
                        crop_x1 = max(0, cb.x1 - mw)
                        crop_y1 = max(0, cb.y1 - mh)
                        crop_x2 = min(img_w, cb.x2 + mw)
                        crop_y2 = min(img_h, cb.y2 + mh)
                        t.best_container_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        _populate_fr_text_crops(t, frame, cb, in_nums, in_isos, frame_idx, img_w, img_h)

                        tracks.append(t)

                if frame_idx % (sample_interval * 4) == 0:
                    pct = 0.05 + 0.45 * (frame_idx / max(1, total_frames))
                    active_count = len([t for t in tracks if t.obs_count >= 3])
                    self._emit_event(
                        OrchestratorEvent(
                            type="PROGRESS",
                            job_id=job_id,
                            data={
                                "stage": "FRONT_REAR",
                                "message": f"Tahap 1/2: Tracking kontainer utuh... ({frame_idx}/{total_frames} frame, {active_count} terdeteksi)",
                                "progress": pct,
                            },
                        )
                    )

            # Cap dilepas segera setelah tracking selesai
            cap.release()

            # -------------------------------------------------------------
            # PHASE 2: CLOUD OCR & STRICT UNIQUE DEDUPLICATION
            # (No duplicate crops, no duplicate DB rows, 100% UTUH crops)
            # -------------------------------------------------------------
            valid_tracks = [t for t in tracks if t.obs_count >= 3 and t.best_container_crop is not None]
            valid_tracks.sort(key=lambda t: t.first_frame_idx)
            total_tracks = len(valid_tracks)
            self.db.update_job_status(job_id, "PROCESSING", total_regions=total_tracks)

            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "FRONT_REAR",
                        "message": f"Tahap 2/2: Seleksi Golden Frame, Cloud OCR & Deduplikasi ({total_tracks} track terdata)...",
                        "progress": 0.52,
                        "log": f"Tracking selesai: {total_tracks} kontainer utuh terdata. Memulai OCR & deduplikasi...",
                    },
                )
            )

            seen_unique_numbers = set()
            container_counter = 0
            tmp_num_path = temp_crop_dir / f"job{job_id}_temp_num.jpg"
            log_cb = audit_logger.on_api_event if audit_logger else None
            actual_cam_pos = "rear" if (camera_position == "rear" or not camera_position) else camera_position

            for idx, t in enumerate(valid_tracks, start=1):
                if self._cancel_requested.is_set():
                    self.db.update_job_status(job_id, "INTERRUPTED", total_regions=container_counter, processed_regions=container_counter)
                    raise RuntimeError("Proses dibatalkan oleh pengguna")

                pct = 0.52 + 0.46 * (idx / max(1, total_tracks))

                if t.best_container_crop is None or t.best_num_crop is None:
                    continue

                # 1. OCR sementara untuk validasi kode unik sebelum menyimpan crop permanen
                imwrite_unicode(tmp_num_path, t.best_num_crop)
                raw_text = ""
                clean_text = ""
                is_valid = False
                try:
                    raw_text, _ = self.ocr_client.recognize_image_file(
                        tmp_num_path,
                        detection_type="container_number",
                        camera_position=actual_cam_pos,
                        log_callback=log_cb,
                        context_info=f"Kontainer #{container_counter + 1} (F{t.best_frame_idx}, FRONT_REAR)",
                    )
                    iso_res = validate_iso6346(raw_text)
                    clean_text = iso_res.clean_number or raw_text.strip().upper()
                    is_valid = iso_res.is_valid
                except Exception as e:
                    raw_text = f"[OCR ERROR: {e}]"
                    clean_text = ""
                    is_valid = False

                # ATURAN STRICT DEDUPLIKASI: Jika nomor kontainer unik ini sudah tersimpan, JANGAN CROP/SIMPAN LAGI!
                if clean_text and len(clean_text) >= 4:
                    if clean_text in seen_unique_numbers:
                        self._emit_event(
                            OrchestratorEvent(
                                type="PROGRESS",
                                job_id=job_id,
                                data={
                                    "stage": "FRONT_REAR",
                                    "message": f"Dilewati (Duplikat): '{clean_text}'",
                                    "progress": min(0.98, pct),
                                    "log": f"[SKIP DUPLICATE] Kontainer '{clean_text}' sudah tersimpan sebelumnya. Mengabaikan duplikasi crop.",
                                },
                            )
                        )
                        continue
                    seen_unique_numbers.add(clean_text)
                else:
                    # Jika OCR kosong, hanya terima jika kontainer benar-benar UTUH dan berdimensi cukup besar
                    c_h, c_w = t.best_container_crop.shape[:2]
                    if not t.is_utuh or c_w < 280 or c_h < 280:
                        continue

                container_counter += 1
                t.clean_text = clean_text
                t.is_valid = is_valid

                # 2. Simpan crop permanen hanya untuk kontainer unik
                num_crop_path = temp_crop_dir / f"job{job_id}_c{container_counter:03d}_num.jpg"
                imwrite_unicode(num_crop_path, t.best_num_crop)

                # Crop ISO jika ada
                iso_text = ""
                iso_crop_path = None
                if t.best_iso_crop is not None and t.best_ib is not None:
                    iso_crop_path = temp_crop_dir / f"job{job_id}_c{container_counter:03d}_iso.jpg"
                    imwrite_unicode(iso_crop_path, t.best_iso_crop)
                    try:
                        it, _ = self.ocr_client.recognize_image_file(
                            iso_crop_path,
                            detection_type="iso_code",
                            camera_position=actual_cam_pos,
                            log_callback=log_cb,
                            context_info=f"Kontainer #{container_counter} ISO (FRONT_REAR)",
                        )
                        iso_text = it.strip().upper()
                        t.iso_text = iso_text
                    except Exception:
                        pass

                # 3. Anotasi visual utuh beresolusi tinggi
                container_crop = t.best_container_crop
                cc_h, cc_w = container_crop.shape[:2]
                annotated_crop = container_crop.copy()

                # Border kontainer warna hijau
                cv2.rectangle(annotated_crop, (2, 2), (cc_w - 3, cc_h - 3), (0, 255, 127), 3)
                label_header = f"Kontainer #{container_counter} (FRONT_REAR)"
                cv2.rectangle(annotated_crop, (2, 2), (min(cc_w - 3, 2 + len(label_header) * 13), 32), (0, 100, 40), -1)
                cv2.putText(
                    annotated_crop,
                    label_header,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                c_box = (t.best_cb.x1, t.best_cb.y1, t.best_cb.x2, t.best_cb.y2) if t.best_cb else (0, 0, cc_w, cc_h)
                mw = max(15, int((c_box[2] - c_box[0]) * 0.04))
                mh = max(15, int((c_box[3] - c_box[1]) * 0.04))
                cx1 = max(0, c_box[0] - mw)
                cy1 = max(0, c_box[1] - mh)

                # Bounding box nomor kontainer warna cyan tepat pada teks
                nb = t.best_nb
                if nb is not None:
                    rel_nx1 = max(0, int(nb.x1) - cx1)
                    rel_ny1 = max(0, int(nb.y1) - cy1)
                    rel_nx2 = min(cc_w, int(nb.x2) - cx1)
                    rel_ny2 = min(cc_h, int(nb.y2) - cy1)
                    cv2.rectangle(annotated_crop, (rel_nx1, rel_ny1), (rel_nx2, rel_ny2), (255, 255, 0), 2)
                    disp_text = clean_text if clean_text else (raw_text if raw_text else "NO_OCR")
                    cv2.rectangle(annotated_crop, (rel_nx1, max(0, rel_ny1 - 24)), (min(cc_w, rel_nx1 + len(disp_text) * 12), rel_ny1), (0, 0, 0), -1)
                    cv2.putText(
                        annotated_crop,
                        disp_text,
                        (rel_nx1 + 2, max(16, rel_ny1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                # Bounding box ISO code warna oranye jika ada
                ib = t.best_ib
                if ib is not None:
                    rel_ix1 = max(0, int(ib.x1) - cx1)
                    rel_iy1 = max(0, int(ib.y1) - cy1)
                    rel_ix2 = min(cc_w, int(ib.x2) - cx1)
                    rel_iy2 = min(cc_h, int(ib.y2) - cy1)
                    cv2.rectangle(annotated_crop, (rel_ix1, rel_iy1), (rel_ix2, rel_iy2), (0, 165, 255), 2)
                    disp_iso = iso_text if iso_text else "ISO"
                    cv2.rectangle(annotated_crop, (rel_ix1, max(0, rel_iy1 - 22)), (min(cc_w, rel_ix1 + len(disp_iso) * 11), rel_iy1), (0, 0, 0), -1)
                    cv2.putText(
                        annotated_crop,
                        disp_iso,
                        (rel_ix1 + 2, max(16, rel_iy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 165, 255),
                        2,
                    )

                clean_tag = clean_text if (clean_text and clean_text != "-") else f"track{t.track_id:02d}"
                iso_tag = f"_{iso_text}" if (iso_text and iso_text != "-") else ""
                base_name = f"{video_stem}_{timestamp_str}_R{container_counter:03d}_{clean_tag}{iso_tag}"

                ann_crop_path = images_dir / f"{base_name}_annotated.jpg"
                raw_crop_path = images_dir / f"{base_name}_raw.jpg"
                imwrite_unicode(ann_crop_path, annotated_crop)
                imwrite_unicode(raw_crop_path, container_crop)

                # 4. Simpan kontainer dan deteksi ke SQLite Database
                c_db_id = self.db.add_container(
                    job_id=job_id,
                    route_order=container_counter,
                    container_id=t.track_id,
                    status="stable",
                    image_path=str(raw_crop_path),
                    annotated_image_path=str(ann_crop_path),
                    pos_x=int(c_box[0]),
                    pos_y=int(c_box[1]),
                    pos_w=int(c_box[2] - c_box[0]),
                    pos_h=int(c_box[3] - c_box[1]),
                    container_type="FRONT_REAR",
                )

                self.db.add_detection(
                    container_id=c_db_id,
                    detection_type="container_number",
                    bbox_x=int(nb.x1) if nb else int(c_box[0]),
                    bbox_y=int(nb.y1) if nb else int(c_box[1]),
                    bbox_w=int(nb.width) if nb else int(c_box[2] - c_box[0]),
                    bbox_h=int(nb.height) if nb else int(c_box[3] - c_box[1]),
                    confidence=float(nb.confidence) if nb else 0.0,
                    crop_image_path=str(num_crop_path) if num_crop_path else str(raw_crop_path),
                    ocr_raw_text=raw_text or "NO_TEXT_DETECTED",
                    ocr_clean_text=clean_text or "-",
                    is_valid_iso=is_valid,
                )

                self.db.add_detection(
                    container_id=c_db_id,
                    detection_type="iso_code",
                    bbox_x=int(t.best_ib.x1) if t.best_ib is not None else 0,
                    bbox_y=int(t.best_ib.y1) if t.best_ib is not None else 0,
                    bbox_w=int(t.best_ib.width) if t.best_ib is not None else 0,
                    bbox_h=int(t.best_ib.height) if t.best_ib is not None else 0,
                    confidence=float(t.best_ib.confidence) if t.best_ib is not None else 0.0,
                    crop_image_path=str(iso_crop_path) if iso_crop_path else None,
                    ocr_raw_text=iso_text or "",
                    ocr_clean_text=iso_text.strip().upper() if iso_text else "-",
                    is_valid_iso=False,
                )

                # 5. Kirim update real-time ke UI
                v_badge = "VALID ISO 6346" if is_valid else ("CHECK DIGIT TIDAK SESUAI" if clean_text and clean_text != "-" else "NO TEXT")
                pct = 0.52 + 0.46 * (idx / max(1, total_tracks))
                self._emit_event(
                    OrchestratorEvent(
                        type="PROGRESS",
                        job_id=job_id,
                        data={
                            "stage": "FRONT_REAR",
                            "message": f"Ditemukan Kontainer #{container_counter} (FRONT_REAR): {clean_text or '-'}",
                            "progress": min(0.98, pct),
                            "log": f"[FRONT_REAR] Kontainer #{container_counter}: '{clean_text or '-'}' ({v_badge})" + (f" | ISO: '{iso_text}'" if iso_text else ""),
                        },
                    )
                )
                self._emit_event(
                    OrchestratorEvent(
                        type="CONTAINER_PROCESSED",
                        job_id=job_id,
                        data={"route_order": container_counter, "processed": container_counter, "total": total_tracks},
                    )
                )

            if tmp_num_path.exists():
                try:
                    tmp_num_path.unlink()
                except Exception:
                    pass
        finally:
            cap.release()
            audit_logger.finalize()

            # Render video hasil anotasi tracking
            out_vid_path = job_output_dir / f"{video_stem}_annotated.mp4"
            try:
                self._render_annotated_video(
                    video_path=video_path,
                    output_video_path=out_vid_path,
                    tracks=valid_tracks,
                    job_id=job_id,
                    mode="FRONT_REAR",
                    C=C_motion,
                )
            except Exception as e:
                self._emit_event(
                    OrchestratorEvent(
                        type="LOG",
                        job_id=job_id,
                        data={"log": f"[WARN] Gagal merender video anotasi: {e}"},
                    )
                )

        self.db.update_job_status(job_id, "PROCESSING", total_regions=container_counter, processed_regions=container_counter)

    def _render_annotated_video(
        self,
        video_path: str,
        output_video_path: Path,
        tracks: List[Any],
        job_id: int,
        mode: str = "FRONT_REAR",
        C: Optional[np.ndarray] = None,
    ) -> None:
        """
        Renders an annotated tracking video with bounding boxes and OCR results.
        Uses cumulative camera motion estimation and multi-anchor warp interpolation
        (annotation_motion.py) for rock-solid tracking glued ('nempel') to container numbers.
        """
        def _on_render_progress(pct: float, msg: str) -> None:
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": msg,
                        "progress": 0.92 + 0.08 * pct,
                        "log": f"[VIDEO EXPORT] {msg}",
                    },
                )
            )

        if C is None:
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": "Menghitung gerak kamera drone per frame...",
                        "progress": 0.85,
                        "log": "[VIDEO EXPORT] Memulai estimasi gerak kamera drone kumulatif...",
                    },
                )
            )

            def _on_motion_progress(pct: float, msg: str) -> None:
                self._emit_event(
                    OrchestratorEvent(
                        type="PROGRESS",
                        job_id=job_id,
                        data={
                            "stage": "VIDEO_EXPORT",
                            "message": msg,
                            "progress": 0.85 + 0.07 * pct,
                            "log": f"[VIDEO EXPORT] {msg}",
                        },
                    )
                )

            C = estimate_cumulative_motion(
                str(video_path),
                cancel=self._cancel_requested.is_set,
                progress_callback=_on_motion_progress,
            )

        def label_fn(t: Any) -> str:
            clean_num = getattr(t, "clean_text", "")
            iso_label = getattr(t, "iso_text", "")
            if clean_num and clean_num != "-":
                full_label = clean_num
                if iso_label and iso_label != "-":
                    full_label += f" | {iso_label}"
                return full_label
            return f"Track #{getattr(t, 'track_id', 1)}"

        overlays = overlays_from_front_rear_tracks(tracks, label_fn, C=C)

        res_mode = getattr(self.config, "video_resolution", "720P").upper()
        if res_mode == "ORIGINAL":
            target_h = None
        elif res_mode == "1080P":
            target_h = 1080
        else:
            target_h = 720

        success = render_overlay_video(
            video_path=str(video_path),
            out_path=str(output_video_path),
            overlays=overlays,
            C=C,
            target_h=target_h,
            cancel=self._cancel_requested.is_set,
            progress_callback=_on_render_progress,
        )

        if success:
            self.db.update_job_output_video(job_id, str(output_video_path))
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": f"Video hasil anotasi berhasil disimpan: {output_video_path.name}",
                        "progress": 1.0,
                        "log": f"[VIDEO EXPORT] File video hasil tersimpan: {output_video_path.name}",
                    },
                )
            )

    def _render_annotated_video_stitching(
        self,
        video_path: str,
        output_video_path: Path,
        tracks_data: List[Dict[str, Any]],
        job_id: int,
        mode: str = "TOP",
    ) -> None:
        """
        Renders an annotated video for stitching modes (TOP / LEFT_RIGHT).
        Uses cumulative camera motion estimation and multi-anchor warp interpolation.
        """
        self._emit_event(
            OrchestratorEvent(
                type="PROGRESS",
                job_id=job_id,
                data={
                    "stage": "VIDEO_EXPORT",
                    "message": "Menghitung gerak kamera drone per frame (stitching)...",
                    "progress": 0.85,
                    "log": "[VIDEO EXPORT] Memulai estimasi gerak kamera drone...",
                },
            )
        )

        def _on_motion_progress(pct: float, msg: str) -> None:
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": msg,
                        "progress": 0.85 + 0.07 * pct,
                        "log": f"[VIDEO EXPORT] {msg}",
                    },
                )
            )

        def _on_render_progress(pct: float, msg: str) -> None:
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": msg,
                        "progress": 0.92 + 0.08 * pct,
                        "log": f"[VIDEO EXPORT] {msg}",
                    },
                )
            )

        has_reg_anchors = any(
            bool(t.get("frame_registrations") or t.get("direct_4k_anchor")) for t in tracks_data
        )
        C: Optional[np.ndarray] = None
        if not has_reg_anchors:
            C = estimate_cumulative_motion(
                str(video_path),
                cancel=self._cancel_requested.is_set,
                progress_callback=_on_motion_progress,
            )

        overlays = overlays_from_stitching_tracks(tracks_data, C=C)

        res_mode = getattr(self.config, "video_resolution", "720P").upper()
        if res_mode == "ORIGINAL":
            target_h = None
        elif res_mode == "1080P":
            target_h = 1080
        else:
            target_h = 720

        success = render_overlay_video(
            video_path=str(video_path),
            out_path=str(output_video_path),
            overlays=overlays,
            C=C,
            target_h=target_h,
            cancel=self._cancel_requested.is_set,
            progress_callback=_on_render_progress,
        )

        if success:
            self.db.update_job_output_video(job_id, str(output_video_path))
            self._emit_event(
                OrchestratorEvent(
                    type="PROGRESS",
                    job_id=job_id,
                    data={
                        "stage": "VIDEO_EXPORT",
                        "message": f"Video hasil anotasi berhasil disimpan: {output_video_path.name}",
                        "progress": 1.0,
                        "log": f"[VIDEO EXPORT] File video hasil tersimpan: {output_video_path.name}",
                    },
                )
            )

    def _handle_manual_override(self, cmd: Dict[str, Any]) -> None:
        detection_id = cmd.get("detection_id")
        container_id = cmd.get("container_id")
        det_type = cmd.get("detection_type") or "container_number"
        text = cmd.get("text", "").strip().upper()
        if not text:
            return
        if det_type != "iso_code" and not detection_id:
            return
        if det_type == "iso_code" and not detection_id and not container_id:
            return

        if det_type == "iso_code":
            is_valid, clean_text, _err = validate_iso_size_type(text)
            text = clean_text or text
        else:
            iso_res = validate_iso6346(text)
            is_valid = iso_res.is_valid

        try:
            # Resolve job_id yang benar untuk UI filter.
            # Prioritas: payload dari UI → _current_job_id → DB lookup.
            event_job_id: Optional[int] = cmd.get("job_id")
            if event_job_id is None:
                event_job_id = self._current_job_id
            if event_job_id is None and detection_id:
                try:
                    with self.db._get_connection() as conn:
                        r = conn.execute(
                            """
                            SELECT c.job_id FROM detections d
                            JOIN containers c ON c.id = d.container_id
                            WHERE d.id = ?
                            """,
                            (int(detection_id),),
                        ).fetchone()
                    if r is not None:
                        event_job_id = int(r["job_id"]) if hasattr(r, "keys") else int(r[0])
                except Exception:
                    event_job_id = None

            if det_type == "iso_code" and not detection_id:
                detection_id = self.db.upsert_iso_detection(
                    container_id=int(container_id),
                    text=text,
                    is_valid_iso=is_valid,
                )
            else:
                self.db.update_detection_manual_override(
                    detection_id=int(detection_id),
                    manual_override_text=text,
                    is_valid_iso=is_valid,
                )
            try:
                self.db.log_audit(
                    "INFO",
                    f"Manual override {det_type} detection {detection_id}: '{text}' (valid: {is_valid})",
                    job_id=event_job_id,
                )
            except Exception:
                pass

            self._emit_event(
                OrchestratorEvent(
                    type="MANUAL_OVERRIDE_SAVED",
                    job_id=int(event_job_id or 0),
                    data={
                        "detection_id": int(detection_id),
                        "detection_type": det_type,
                        "text": text,
                        "is_valid": is_valid,
                    },
                )
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                self._emit_event(
                    OrchestratorEvent(
                        type="LOG",
                        job_id=self._current_job_id or 0,
                        data={"log": f"[MANUAL OVERRIDE ERROR] det_id={detection_id}: {e}\n{tb}"},
                    )
                )
            except Exception:
                print(f"[FATAL] Manual override gagal: {e}\n{tb}")

    def _handle_re_ocr_crop(self, cmd: Dict[str, Any]) -> None:
        det_id = cmd.get("detection_id")
        container_id = cmd.get("container_id")
        job_id = cmd.get("job_id") or self._current_job_id
        img_path = cmd.get("image_path")
        det_type = cmd.get("detection_type") or "container_number"
        cam_pos = cmd.get("camera_position") or "rear"

        if not img_path or not Path(img_path).is_file():
            return
        if not det_id and not (det_type == "iso_code" and container_id):
            return

        try:
            # 1. Panggil Halotec OCR API
            raw_text, confidence = self.ocr_client.recognize_image_file(
                Path(img_path),
                detection_type=det_type,
                camera_position=cam_pos,
                req_id=job_id or 1,
                context_info=f"Re-OCR Manual (Det ID: {det_id})",
            )

            # 2. Validasi & bersihkan teks
            is_valid = False
            clean_text = raw_text.strip().upper()

            if det_type == "container_number":
                iso_res = validate_iso6346(raw_text)
                is_valid = iso_res.is_valid
                clean_text = iso_res.clean_number if iso_res.clean_number else clean_text
            elif det_type == "iso_code":
                is_valid, clean_text, _err = validate_iso_size_type(raw_text)

            if det_type == "iso_code" and not is_valid:
                # Jangan simpan hasil yang bukan kode ISO (mis. nomor kontainer) ke kolom ISO.
                self._emit_event(
                    OrchestratorEvent(
                        type="CONTAINER_PROCESSED",
                        job_id=int(job_id or 0),
                        data={
                            "reocr": True,
                            "success": False,
                            "message": f"Re-OCR ISO gagal: '{raw_text or '-'}' bukan kode ISO valid",
                        },
                    )
                )
                return
            if not clean_text:
                self._emit_event(
                    OrchestratorEvent(
                        type="CONTAINER_PROCESSED",
                        job_id=int(job_id or 0),
                        data={"reocr": True, "success": False, "message": "Re-OCR tidak menghasilkan teks"},
                    )
                )
                return

            # 3. Update Database (buat baris ISO baru bila belum ada)
            if not det_id and det_type == "iso_code":
                det_id = self.db.add_detection(
                    container_id=int(container_id),
                    detection_type="iso_code",
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=0,
                    bbox_h=0,
                    confidence=float(confidence),
                    crop_image_path=str(img_path),
                    ocr_raw_text=raw_text,
                    ocr_clean_text=clean_text,
                    is_valid_iso=is_valid,
                )
            else:
                self.db.update_detection_after_reocr(
                    detection_id=int(det_id),
                    new_crop_path=str(img_path),
                    raw_text=raw_text,
                    clean_text=clean_text,
                    is_valid_iso=is_valid,
                    confidence=float(confidence),
                )

            # 4. Kirim event ke UI agar tabel refresh
            self._emit_event(
                OrchestratorEvent(
                    type="CONTAINER_PROCESSED",
                    job_id=int(job_id or 0),
                    data={"message": f"Re-OCR sukses untuk deteksi #{det_id}", "reocr": True},
                )
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._emit_event(
                OrchestratorEvent(
                    type="LOG",
                    job_id=int(job_id or 0),
                    data={"log": f"[RE-OCR ERROR] det_id={det_id}: {e}\n{tb}"},
                )
            )
            self._emit_event(
                OrchestratorEvent(
                    type="CONTAINER_PROCESSED",
                    job_id=int(job_id or 0),
                    data={"reocr": True, "success": False, "message": f"Re-OCR gagal: {e}"},
                )
            )
    
    def _handle_update_config(self, cmd: Dict[str, Any]) -> None:
        new_cfg_dict = cmd.get("config", {})
        if new_cfg_dict:
            self.config = AppConfig.from_dict(new_cfg_dict)
            self.config.save()
            self._detector = None  # Force re-init with updated compute device/thresholds
            self.stitching_adapter.config = self.config
            self.ocr_client = HalotecOcrClient(
                base_url=self.config.api.url,
                gate=self.config.api.gate,
                timeout=self.config.api.timeout_seconds,
                max_retries=self.config.api.max_retries,
            )
