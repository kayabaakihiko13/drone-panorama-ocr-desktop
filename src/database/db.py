"""
SQLite Database Layer (Single-Writer Architecture).
Follows TDD_Container_OCR_Windows_v1.9.md:
- Only Orchestrator writes to the database.
- Read operations are lightweight and WAL-mode enabled.
- Interrupted jobs are recovered on startup.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.config import BASE_DIR

DEFAULT_DB_PATH = BASE_DIR / "data" / "app.db"


class Database:
    """Thread-safe single-writer SQLite wrapper for Container OCR Desktop."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL, -- QUEUED, PROCESSING, COMPLETED, FAILED, INTERRUPTED
                    total_regions INTEGER DEFAULT 0,
                    processed_regions INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS containers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    route_order INTEGER NOT NULL,
                    container_id INTEGER NOT NULL,
                    status TEXT NOT NULL, -- stable, unstable
                    container_type TEXT DEFAULT 'LEFT_RIGHT', -- LEFT_RIGHT, FRONT_REAR, TOP
                    image_path TEXT NOT NULL,
                    annotated_image_path TEXT,
                    pos_x INTEGER DEFAULT 0,
                    pos_y INTEGER DEFAULT 0,
                    pos_w INTEGER DEFAULT 0,
                    pos_h INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container_id INTEGER NOT NULL,
                    detection_type TEXT NOT NULL, -- iso_code, container_number
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    crop_image_path TEXT,
                    ocr_raw_text TEXT,
                    ocr_clean_text TEXT,
                    is_valid_iso INTEGER DEFAULT 0,
                    manual_override_text TEXT,
                    is_reviewed INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (container_id) REFERENCES containers (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_containers_job ON containers(job_id);
                CREATE INDEX IF NOT EXISTS idx_detections_container ON detections(container_id);
                """
            )
            try:
                conn.execute("ALTER TABLE containers ADD COLUMN container_type TEXT DEFAULT 'LEFT_RIGHT'")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN output_video_path TEXT")
            except Exception:
                pass

    def mark_interrupted_jobs(self) -> int:
        """Mark any jobs left in PROCESSING status as INTERRUPTED on application startup."""
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'INTERRUPTED', updated_at = ?
                WHERE status IN ('PROCESSING', 'QUEUED')
                """,
                (now,),
            )
            return cursor.rowcount

    def create_job(self, name: str, video_path: str, mode: str) -> int:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (name, video_path, mode, status, created_at, updated_at)
                VALUES (?, ?, ?, 'QUEUED', ?, ?)
                """,
                (name, video_path, mode, now, now),
            )
            return cursor.lastrowid  # type: ignore

    def update_job_status(
        self,
        job_id: int,
        status: str,
        total_regions: Optional[int] = None,
        processed_regions: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            updates = ["status = ?", "updated_at = ?"]
            params: List[Any] = [status, now]

            if total_regions is not None:
                updates.append("total_regions = ?")
                params.append(total_regions)
            if processed_regions is not None:
                updates.append("processed_regions = ?")
                params.append(processed_regions)
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            params.append(job_id)
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, params)

    def update_job_output_video(self, job_id: int, output_video_path: str) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET output_video_path = ?, updated_at = ? WHERE id = ?",
                (output_video_path, now, job_id),
            )

    def add_container(
        self,
        job_id: int,
        route_order: int,
        container_id: int,
        status: str,
        image_path: str,
        annotated_image_path: Optional[str] = None,
        pos_x: int = 0,
        pos_y: int = 0,
        pos_w: int = 0,
        pos_h: int = 0,
        container_type: str = "LEFT_RIGHT",
    ) -> int:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO containers (
                    job_id, route_order, container_id, status, container_type, image_path,
                    annotated_image_path, pos_x, pos_y, pos_w, pos_h, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    route_order,
                    container_id,
                    status,
                    container_type,
                    image_path,
                    annotated_image_path,
                    pos_x,
                    pos_y,
                    pos_w,
                    pos_h,
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def update_container_image(
        self,
        job_id: int,
        route_order: int,
        image_path: str,
        annotated_image_path: Optional[str] = None,
        status: str = "completed",
        container_type: Optional[str] = None,
    ) -> None:
        """Updates stitched panorama image paths for a container asynchronously."""
        with self._lock, self._get_connection() as conn:
            if container_type:
                conn.execute(
                    """
                    UPDATE containers
                    SET image_path = ?, annotated_image_path = ?, status = ?, container_type = ?
                    WHERE job_id = ? AND route_order = ?
                    """,
                    (image_path, annotated_image_path, status, container_type, job_id, route_order),
                )
            else:
                conn.execute(
                    """
                    UPDATE containers
                    SET image_path = ?, annotated_image_path = ?, status = ?
                    WHERE job_id = ? AND route_order = ?
                    """,
                    (image_path, annotated_image_path, status, job_id, route_order),
                )

    def add_detection(
        self,
        container_id: int,
        detection_type: str,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        confidence: float,
        crop_image_path: Optional[str] = None,
        ocr_raw_text: Optional[str] = None,
        ocr_clean_text: Optional[str] = None,
        is_valid_iso: bool = False,
    ) -> int:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detections (
                    container_id, detection_type, bbox_x, bbox_y, bbox_w, bbox_h,
                    confidence, crop_image_path, ocr_raw_text, ocr_clean_text,
                    is_valid_iso, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    container_id,
                    detection_type,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    confidence,
                    crop_image_path,
                    ocr_raw_text,
                    ocr_clean_text,
                    1 if is_valid_iso else 0,
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def update_detection_ocr(
        self,
        detection_id: int,
        crop_image_path: str,
        ocr_raw_text: str,
        ocr_clean_text: str,
        is_valid_iso: bool,
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE detections
                SET crop_image_path = ?, ocr_raw_text = ?, ocr_clean_text = ?, is_valid_iso = ?
                WHERE id = ?
                """,
                (crop_image_path, ocr_raw_text, ocr_clean_text, 1 if is_valid_iso else 0, detection_id),
            )
    def update_detection_after_reocr(
        self,
        detection_id: int,
        new_crop_path: str,
        raw_text: str,
        clean_text: str,
        is_valid_iso: bool,
        confidence: float,
    ) -> None:
        """Mengupdate hasil deteksi setelah proses Re-OCR (rotasi gambar)."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE detections
                SET crop_image_path = ?, ocr_raw_text = ?, ocr_clean_text = ?, 
                    is_valid_iso = ?, confidence = ?, is_reviewed = 0,
                    manual_override_text = NULL
                WHERE id = ?
                """,
                (
                    new_crop_path,
                    raw_text,
                    clean_text,
                    1 if is_valid_iso else 0,
                    confidence,
                    detection_id,
                ),
            )
    def update_detection_manual_override(
        self,
        detection_id: int,
        manual_override_text: str,
        is_valid_iso: bool,
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE detections
                SET manual_override_text = ?, is_valid_iso = ?, is_reviewed = 1
                WHERE id = ?
                """,
                (manual_override_text, 1 if is_valid_iso else 0, detection_id),
            )

    def upsert_iso_detection(
        self,
        container_id: int,
        text: str,
        is_valid_iso: bool,
        crop_image_path: Optional[str] = None,
        confidence: float = 0.0,
    ) -> int:
        """Creates or updates the iso_code detection row for a container.

        Used by manual correction / re-OCR when the ISO code was never detected,
        so the operator can still record a value.
        """
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM detections WHERE container_id = ? AND detection_type = 'iso_code' ORDER BY id ASC LIMIT 1",
                (container_id,),
            ).fetchone()
            if row is not None:
                detection_id = int(row["id"] if hasattr(row, "keys") else row[0])
                conn.execute(
                    """
                    UPDATE detections
                    SET manual_override_text = ?, ocr_clean_text = ?, ocr_raw_text = COALESCE(ocr_raw_text, ?),
                        is_valid_iso = ?, is_reviewed = 1
                    WHERE id = ?
                    """,
                    (text, text, text, 1 if is_valid_iso else 0, detection_id),
                )
                return detection_id

            cursor = conn.execute(
                """
                INSERT INTO detections (
                    container_id, detection_type, bbox_x, bbox_y, bbox_w, bbox_h,
                    confidence, crop_image_path, ocr_raw_text, ocr_clean_text,
                    is_valid_iso, manual_override_text, is_reviewed, created_at
                )
                VALUES (?, 'iso_code', 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    container_id,
                    confidence,
                    crop_image_path,
                    text,
                    text,
                    1 if is_valid_iso else 0,
                    text,
                    now,
                ),
            )
            return int(cursor.lastrowid)  # type: ignore

    def log_audit(self, level: str, message: str, job_id: Optional[int] = None) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (job_id, level, message, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, level, message, now),
            )

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_containers_for_job(self, job_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM containers WHERE job_id = ? ORDER BY route_order ASC",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_detections_for_container(self, container_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE container_id = ? ORDER BY id ASC",
                (container_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_full_container_view(self, job_id: int) -> List[Dict[str, Any]]:
        """Returns containers enriched with their container_number and iso_code detections."""
        with self._get_connection() as conn:
            containers = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM containers WHERE job_id = ? ORDER BY route_order ASC",
                    (job_id,),
                ).fetchall()
            ]

            for c in containers:
                detections = [
                    dict(d)
                    for d in conn.execute(
                        "SELECT * FROM detections WHERE container_id = ?",
                        (c["id"],),
                    ).fetchall()
                ]
                c["detections"] = {d["detection_type"]: d for d in detections}
            return containers
