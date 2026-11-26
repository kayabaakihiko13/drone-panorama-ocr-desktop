"""
Main Application Window (PySide6).
Follows TDD_Container_OCR_Windows_v1.9.md:
- UI thread does NOT execute AI models, stitching, or synchronous DB writes.
- In-process QTimer pulls events from Orchestrator's event_queue.
- Displays Job history, Container results, ISO 6346 validation badges, RAW Panoramas, and zoomed ROI crops.
- Provides immediate manual review and correction interface.
"""
from __future__ import annotations

import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config import BASE_DIR, AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator, OrchestratorEvent
from src.recognition.detector import describe_compute_plan
from src.ui.settings_dialog import SettingsDialog
from src.ui.toast import ElidedLabel, ToastManager
from src.ui.video_player_dialog import VideoPlayerDialog
from src.utils.iso6346 import validate_iso6346


class MainWindow(QMainWindow):
    def __init__(
        self,
        cmd_queue: queue.Queue,
        evt_queue: queue.Queue,
        config: AppConfig,
        db: Database,
        orchestrator: Optional[Orchestrator] = None,
    ):
        super().__init__()
        self.cmd_queue = cmd_queue
        self.evt_queue = evt_queue
        self.config = config
        self.db = db
        self.orchestrator = orchestrator

        self.current_job_id: Optional[int] = None
        self.selected_container_id: Optional[int] = None
        self.selected_detection_id: Optional[int] = None
        self.is_job_running = False

        self.setWindowTitle("Halotec Container OCR - Drone View (Windows Desktop)")
        self.resize(1300, 850)

        self._init_ui()
        self._toasts = ToastManager(self.centralWidget())
        self._init_event_timer()
        self._load_jobs()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 1. Top Control Bar
        top_bar = QHBoxLayout()

        self.btn_new_job = QPushButton("📁 Pilih Video Drone...")
        self.btn_new_job.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.btn_new_job.clicked.connect(self._on_select_video)
        top_bar.addWidget(self.btn_new_job)

        self.btn_play_input = QPushButton("▶ Putar Video Input")
        self.btn_play_input.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.btn_play_input.setEnabled(False)
        self.btn_play_input.setToolTip("Putar video drone input yang sedang dipilih")
        self.btn_play_input.clicked.connect(self._on_play_input_video)
        top_bar.addWidget(self.btn_play_input)

        top_bar.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("LEFT_RIGHT (Samping Drone - Stitching)", "LEFT_RIGHT")
        self.mode_combo.addItem("TOP (Atas Drone - Stitching)", "TOP")
        self.mode_combo.addItem("FRONT_REAR (Depan/Belakang - Langsung Frame)", "FRONT_REAR")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top_bar.addWidget(self.mode_combo)

        top_bar.addWidget(QLabel("Posisi Sensor:"))
        self.cam_pos_combo = QComboBox()
        top_bar.addWidget(self.cam_pos_combo)
        self._update_cam_pos_options("LEFT_RIGHT")

        self.btn_start = QPushButton("Mulai Proses")
        self.btn_start.setStyleSheet("background-color: #0078D4; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        self.btn_start.clicked.connect(self._on_start_or_cancel_job)
        top_bar.addWidget(self.btn_start)

        top_bar.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(260)
        top_bar.addWidget(self.progress_bar)

        self.lbl_status = ElidedLabel("Siap")
        self.lbl_status.setStyleSheet("color: #0078D4; font-weight: bold;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_status.setMinimumWidth(0)
        top_bar.addWidget(self.lbl_status, 1)

        self.btn_settings = QPushButton("Pengaturan")
        self.btn_settings.clicked.connect(self._open_settings)
        top_bar.addWidget(self.btn_settings)

        self.btn_logout = QPushButton("Keluar (Logout)")
        self.btn_logout.setStyleSheet("color: #d83b01;")
        self.btn_logout.clicked.connect(self._on_logout)
        top_bar.addWidget(self.btn_logout)

        main_layout.addLayout(top_bar)

        # 2. Main Content Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Job History List and Live Activity Log
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("<b>Riwayat Job Pemrosesan:</b>"))

        self.jobs_table = QTableWidget()
        self.jobs_table.setColumnCount(4)
        self.jobs_table.setHorizontalHeaderLabels(["ID", "Nama Video", "Mode", "Status"])
        self.jobs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.jobs_table.setMaximumHeight(180)
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.jobs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.jobs_table.itemSelectionChanged.connect(self._on_job_selected)
        left_layout.addWidget(self.jobs_table)

        left_layout.addSpacing(6)
        left_layout.addWidget(QLabel("<b>Log Aktivitas & Estimasi Proses:</b>"))
        self.txt_activity_log = QPlainTextEdit()
        self.txt_activity_log.setReadOnly(True)
        self.txt_activity_log.setStyleSheet(
            "background-color: #121212; color: #00FF7F; font-family: Consolas, 'Courier New', monospace; "
            "font-size: 11px; border: 1px solid #333; border-radius: 4px; padding: 4px;"
        )
        self.txt_activity_log.setMinimumHeight(240)
        left_layout.addWidget(self.txt_activity_log)
        main_splitter.addWidget(left_widget)

        # Right: Container Table and Inspection View
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_header_lay = QHBoxLayout()
        right_header_lay.addWidget(QLabel("<b>Hasil Deteksi & Pengenalan Kontainer (ISO 6346):</b>"))
        self.lbl_container_summary = QLabel("")
        self.lbl_container_summary.setStyleSheet("color: #888; font-size: 11px; margin-left: 8px;")
        right_header_lay.addWidget(self.lbl_container_summary)
        right_header_lay.addStretch()

        self.btn_play_output = QPushButton("Putar Video Hasil")
        self.btn_play_output.setStyleSheet("padding: 3px 8px; font-size: 11px; font-weight: bold; background-color: #005a9e; color: white;")
        self.btn_play_output.setEnabled(False)
        self.btn_play_output.setToolTip("Putar video hasil tracking & anotasi kontainer (.mp4)")
        self.btn_play_output.clicked.connect(self._on_play_output_video)
        right_header_lay.addWidget(self.btn_play_output)

        self.btn_open_folder = QPushButton("Save Video Hasil")
        self.btn_open_folder.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.btn_open_folder.clicked.connect(self._on_open_output_folder)
        right_header_lay.addWidget(self.btn_open_folder)

        self.btn_export_csv = QPushButton("Ekspor CSV")
        self.btn_export_csv.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        right_header_lay.addWidget(self.btn_export_csv)

        right_layout.addLayout(right_header_lay)

        self.container_table = QTableWidget()
        self.container_table.setColumnCount(8)
        self.container_table.setHorizontalHeaderLabels([
            "Region", "Container ID", "Status Spasial", "Nomor Kontainer (OCR)",
            "ISO 6346 Check", "ISO Code", "Confidence", "Manual Review"
        ])
        self.container_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.container_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.container_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.container_table.itemSelectionChanged.connect(self._on_container_selected)
        right_layout.addWidget(self.container_table)

        # Bottom Inspection & Preview Splitter
        preview_group = QGroupBox("Detail Visual & Verifikasi Lapangan (Zero Double-Detection)")
        preview_layout = QVBoxLayout(preview_group)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panorama / Container Image View
        pano_box = QWidget()
        pano_lay = QVBoxLayout(pano_box)
        pano_lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_pano_header = QLabel("Citra Kontainer Terdeteksi (RAW / Anotasi Bounding Box):")
        pano_lay.addWidget(self.lbl_pano_header)
        self.lbl_pano_img = QLabel("Pilih baris kontainer untuk menampilkan citra")
        self.lbl_pano_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_pano_img.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444;")
        self.lbl_pano_img.setMinimumHeight(240)
        pano_lay.addWidget(self.lbl_pano_img)
        preview_splitter.addWidget(pano_box)

        # Cropped ROI View (Container Number & ISO Code)
        crop_box = QWidget()
        crop_lay = QVBoxLayout(crop_box)
        crop_lay.setContentsMargins(0, 0, 0, 0)
        crop_lay.addWidget(QLabel("Crop Resolusi Penuh (Dikirim ke Halotec OCR API):"))

        crop_tabs = QTabWidget()

        # Tab Container Number Crop
        tab_num = QWidget()
        tab_num_lay = QVBoxLayout(tab_num)
        self.lbl_crop_num = QLabel("Crop Nomor Kontainer")
        self.lbl_crop_num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_num.setStyleSheet("background-color: #252526; color: #888;")
        tab_num_lay.addWidget(self.lbl_crop_num)
        crop_tabs.addTab(tab_num, "Nomor Kontainer")

        # Tab ISO Code Crop
        tab_iso = QWidget()
        tab_iso_lay = QVBoxLayout(tab_iso)
        self.lbl_crop_iso = QLabel("Crop ISO Code")
        self.lbl_crop_iso.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_iso.setStyleSheet("background-color: #252526; color: #888;")
        tab_iso_lay.addWidget(self.lbl_crop_iso)
        crop_tabs.addTab(tab_iso, "ISO Code")

        crop_lay.addWidget(crop_tabs)
        preview_splitter.addWidget(crop_box)
        preview_layout.addWidget(preview_splitter)

        # Quick Manual Correction / Review Bar
        review_bar = QHBoxLayout()
        review_bar.addWidget(QLabel("<b>Koreksi Manual:</b>"))
        self.edit_manual_num = QLineEdit()
        self.edit_manual_num.setPlaceholderText("Edit nomor kontainer...")
        self.edit_manual_num.textChanged.connect(self._on_manual_text_changed)
        review_bar.addWidget(self.edit_manual_num)

        self.lbl_iso_feedback = QLabel("-")
        review_bar.addWidget(self.lbl_iso_feedback)

        self.btn_save_override = QPushButton("Simpan Koreksi")
        self.btn_save_override.setStyleSheet("background-color: #107C41; color: white; font-weight: bold; padding: 4px 12px;")
        self.btn_save_override.clicked.connect(self._on_save_manual_override)
        review_bar.addWidget(self.btn_save_override)

        preview_layout.addLayout(review_bar)
        right_layout.addWidget(preview_group)

        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 7)
        main_layout.addWidget(main_splitter)

        self.selected_video_path: Optional[str] = None

    def _init_event_timer(self) -> None:
        """Polls orchestrator events on the Qt UI thread."""
        self.event_timer = QTimer(self)
        self.event_timer.setInterval(100)
        self.event_timer.timeout.connect(self._process_orchestrator_events)
        self.event_timer.start()

    def _process_orchestrator_events(self) -> None:
        while not self.evt_queue.empty():
            try:
                evt: OrchestratorEvent = self.evt_queue.get_nowait()
            except queue.Empty:
                break

            now_str = datetime.now().strftime("%H:%M:%S")

            if evt.type == "JOB_STARTED":
                self.current_job_id = evt.job_id
                self._clear_workspace_for_new_job()
                self._set_start_button_state(True)
                self.show_toast(f"Job {evt.job_id} dimulai...", "info")
                self.progress_bar.setValue(2)
                self.txt_activity_log.setPlainText(f"[{now_str}] === MULAI JOB {evt.job_id}: {evt.data.get('name')} ===\n")
                self.txt_activity_log.appendPlainText(f"[{now_str}] Mode: {evt.data.get('mode')}, Sensor: {evt.data.get('camera_position')}")
                self.txt_activity_log.ensureCursorVisible()
                self._load_jobs()
            elif evt.type == "PROGRESS":
                pct = int(evt.data.get("progress", 0.0) * 100)
                self.progress_bar.setValue(pct)
                msg = evt.data.get("message", "Memproses...")
                self.lbl_status.setText(msg)

                log_entry = evt.data.get("log")
                if log_entry:
                    self.txt_activity_log.appendPlainText(f"[{now_str}] {log_entry}")
                    self.txt_activity_log.ensureCursorVisible()
                elif msg:
                    if not msg.startswith("Tahap 1/4: Tracking frame") or pct % 10 == 0:
                        self.txt_activity_log.appendPlainText(f"[{now_str}] {msg}")
                        self.txt_activity_log.ensureCursorVisible()
            elif evt.type == "CONTAINER_PROCESSED":
                if self.current_job_id == evt.job_id:
                    self._load_containers_for_job(evt.job_id)
            elif evt.type == "JOB_COMPLETED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(100)
                self.show_toast("Job Selesai.", "success")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [SUKSES] {evt.data.get('message', 'Job selesai!')}\n")
                self.txt_activity_log.ensureCursorVisible()
                self._load_jobs()
                if self.current_job_id == evt.job_id:
                    self._load_containers_for_job(evt.job_id)
            elif evt.type == "JOB_FAILED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(0)
                self.show_toast("Job Gagal.", "error")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [GAGAL] {evt.data.get('error')}\n")
                self.txt_activity_log.ensureCursorVisible()
                self._load_jobs()
                QMessageBox.critical(self, "Error Job", f"Job {evt.job_id} gagal:\n{evt.data.get('error')}")
            elif evt.type == "JOB_CANCELLED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(0)
                self.show_toast("Proses Dibatalkan.", "warning")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [DIBATALKAN] {evt.data.get('message', 'Proses dibatalkan oleh pengguna.')}\n")
                self.txt_activity_log.ensureCursorVisible()
                self._load_jobs()
            elif evt.type == "MANUAL_OVERRIDE_SAVED":
                if self.current_job_id:
                    self._load_containers_for_job(self.current_job_id)
            elif evt.type == "LOG":
                log_entry = evt.data.get("log") or evt.data.get("message")
                if log_entry:
                    if not log_entry.startswith("["):
                        log_entry = f"[{now_str}] {log_entry}"
                    self.txt_activity_log.appendPlainText(log_entry)
                    self.txt_activity_log.ensureCursorVisible()

    def _on_select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih File Video Drone",
            str(BASE_DIR),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)",
        )
        if path:
            self.selected_video_path = path
            self.btn_new_job.setText(f"📁 {Path(path).name}")
            self.show_toast(f"Video dipilih: {Path(path).name}", "info")
            self.btn_play_input.setEnabled(True)

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self._update_cam_pos_options(mode)

    def _update_cam_pos_options(self, mode: str) -> None:
        self.cam_pos_combo.clear()
        if mode == "LEFT_RIGHT":
            self.cam_pos_combo.addItem("Sisi Kiri Drone (LEFT)", "left")
            self.cam_pos_combo.addItem("Sisi Kanan Drone (RIGHT)", "right")
        elif mode == "TOP":
            self.cam_pos_combo.addItem("Atas Kontainer (TOP)", "top")
        elif mode == "FRONT_REAR":
            self.cam_pos_combo.addItem("Belakang Kontainer (REAR)", "rear")
            self.cam_pos_combo.addItem("Depan Kontainer (FRONT)", "front")

    def _set_start_button_state(self, running: bool) -> None:
        self.is_job_running = running
        if running:
            self.btn_start.setText("Batalkan Proses")
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet(
                "background-color: #D83B01; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;"
            )
        else:
            self.btn_start.setText("Mulai Proses")
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet(
                "background-color: #0078D4; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;"
            )

    def _on_start_or_cancel_job(self) -> None:
        if self.is_job_running:
            reply = QMessageBox.question(
                self,
                "Konfirmasi Pembatalan",
                "Apakah Anda yakin ingin membatalkan proses pemrosesan video yang sedang berjalan?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.btn_start.setEnabled(False)
                self.btn_start.setText("Membatalkan...")
                self.lbl_status.setText("Membatalkan proses...")
                if self.orchestrator:
                    self.orchestrator.cancel_current_job()
                self.cmd_queue.put({"action": "CANCEL_JOB"})
        else:
            self._on_start_job()

    def _clear_workspace_for_new_job(self) -> None:
        """Cleans all previous job results, inspection views, and logs cleanly for a fresh start."""
        self.container_table.setRowCount(0)
        self.txt_activity_log.clear()
        self.lbl_pano_img.setText("Memproses video... Citra kontainer akan tampil otomatis begitu terdeteksi.")
        self.lbl_pano_img.setPixmap(QPixmap())
        self.lbl_crop_num.setText("Menunggu deteksi...")
        self.lbl_crop_num.setPixmap(QPixmap())
        self.lbl_crop_iso.setText("Menunggu deteksi...")
        self.lbl_crop_iso.setPixmap(QPixmap())
        self.edit_manual_num.clear()
        self.lbl_iso_feedback.setText("-")
        self.btn_save_override.setEnabled(False)
        self.selected_container_id = None
        self.selected_detection_id = None
        self.lbl_container_summary.setText("Memulai proses baru...")
        self.lbl_container_summary.setStyleSheet("color: #0078D4; font-size: 11px; margin-left: 8px;")
        self.btn_play_output.setEnabled(False)

    def _on_start_job(self) -> None:
        if not self.selected_video_path:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file video drone terlebih dahulu!")
            return

        mode = self.mode_combo.currentData()
        camera_position = self.cam_pos_combo.currentData() or "left"
        name = Path(self.selected_video_path).stem

        # Pre-run hardware confirmation so user sees active GPU/CPU per stage before running
        while True:
            req_dev = str(getattr(self.config.stitching, "device", "auto") or "auto")
            plan = describe_compute_plan(req_dev)
            summary_lines = [
                f"Video: {Path(self.selected_video_path).name} | Mode: {mode} (Sensor: {camera_position})",
                f"Pengaturan Komputasi: {req_dev.upper()}",
                "",
                f"• Tracking Frame & Deteksi ONNX: {plan['track_provider']}",
            ]
            if mode in ("LEFT_RIGHT", "TOP"):
                summary_lines.append(f"• Stitching Panorama (ALIKED/LightGlue): {plan['stitch_device']}")
            if plan["hint"]:
                summary_lines.extend(["", f"Catatan: {plan['hint']}"])

            confirm_box = QMessageBox(self)
            confirm_box.setWindowTitle("Konfirmasi Perangkat & Mulai Proses")
            confirm_box.setIcon(
                QMessageBox.Icon.Information if plan["uses_gpu"] else QMessageBox.Icon.Warning
            )
            confirm_box.setText("\n".join(summary_lines))
            btn_go = confirm_box.addButton("Mulai Proses", QMessageBox.ButtonRole.AcceptRole)
            btn_cfg = confirm_box.addButton("Ubah di Pengaturan...", QMessageBox.ButtonRole.ActionRole)
            btn_cancel = confirm_box.addButton("Batal", QMessageBox.ButtonRole.RejectRole)
            confirm_box.setDefaultButton(btn_go)
            confirm_box.exec()

            clicked = confirm_box.clickedButton()
            if clicked == btn_cfg:
                self._open_settings()
                continue
            if clicked != btn_go or clicked == btn_cancel:
                return
            break

        self._clear_workspace_for_new_job()
        self.cmd_queue.put({
            "action": "START_JOB",
            "name": name,
            "video_path": self.selected_video_path,
            "mode": mode,
            "camera_position": camera_position,
        })

    def _load_jobs(self) -> None:
        jobs = self.db.get_all_jobs()
        self.jobs_table.setRowCount(len(jobs))

        for row, job in enumerate(jobs):
            self.jobs_table.setItem(row, 0, QTableWidgetItem(str(job["id"])))
            self.jobs_table.setItem(row, 1, QTableWidgetItem(job["name"]))
            self.jobs_table.setItem(row, 2, QTableWidgetItem(job["mode"]))

            status_item = QTableWidgetItem(job["status"])
            if job["status"] == "COMPLETED":
                status_item.setForeground(QColor("#107C41"))
            elif job["status"] == "PROCESSING":
                status_item.setForeground(QColor("#0078D4"))
            elif job["status"] == "FAILED":
                status_item.setForeground(QColor("#D83B01"))
            elif job["status"] == "INTERRUPTED":
                status_item.setForeground(QColor("#FF8C00"))
            self.jobs_table.setItem(row, 3, status_item)

        if jobs and self.current_job_id is None:
            self.jobs_table.selectRow(0)

    def _on_job_selected(self) -> None:
        selected_rows = self.jobs_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        job_id_item = self.jobs_table.item(row, 0)
        job_name_item = self.jobs_table.item(row, 1)
        job_status_item = self.jobs_table.item(row, 3)
        if job_id_item:
            self.current_job_id = int(job_id_item.text())
            self._load_containers_for_job(self.current_job_id)
            if not self.is_job_running:
                j_name = job_name_item.text() if job_name_item else ""
                j_status = job_status_item.text() if job_status_item else ""
                cnt = self.container_table.rowCount()
                job_dir = self._resolve_job_output_dir(self.current_job_id)
                log_file = (job_dir / "logs" / "job_activity.log") if job_dir else None
                if log_file and log_file.is_file():
                    try:
                        full_log = log_file.read_text(encoding="utf-8", errors="ignore")
                        self.txt_activity_log.setPlainText(full_log)
                        sb = self.txt_activity_log.verticalScrollBar()
                        sb.setValue(sb.maximum())
                        return
                    except Exception:
                        pass
                self.txt_activity_log.setPlainText(
                    f"=== Riwayat Job #{self.current_job_id}: {j_name} ({j_status}) ===\n"
                    f"Total Kontainer Terdeteksi: {cnt}\n\n"
                    f"• Klik baris kontainer di atas untuk melihat Citra RAW, Anotasi, dan Crop ROI.\n"
                    f"• Gunakan tombol '📁 Buka Folder Hasil' untuk membuka file citra di Windows Explorer.\n"
                    f"• Gunakan tombol '📥 Ekspor CSV' untuk mengunduh rekap hasil pengenalan kontainer."
                )

    def _load_containers_for_job(self, job_id: int) -> None:
        out_vid = self._get_job_output_video(job_id)
        self.btn_play_output.setEnabled(bool(out_vid and Path(out_vid).is_file()))

        containers = self.db.get_full_container_view(job_id)
        self.container_table.setRowCount(len(containers))

        total_cnt = len(containers)
        valid_cnt = sum(1 for c in containers if c.get("detections", {}).get("container_number", {}).get("is_valid_iso"))
        invalid_cnt = total_cnt - valid_cnt

        if total_cnt > 0:
            self.lbl_container_summary.setText(f"Total: {total_cnt} | Valid ISO: {valid_cnt} | Perlu Review: {invalid_cnt}")
            self.lbl_container_summary.setStyleSheet(
                "color: #107C41; font-weight: bold; background-color: #1b3323; border: 1px solid #107C41; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; margin-left: 8px;"
            )
        else:
            self.lbl_container_summary.setText("Belum ada kontainer terdeteksi")
            self.lbl_container_summary.setStyleSheet("color: #888; font-size: 11px; margin-left: 8px;")

        for row, c in enumerate(containers):
            self.container_table.setItem(row, 0, QTableWidgetItem(f"R{c['route_order']:03d}"))
            self.container_table.setItem(row, 1, QTableWidgetItem(str(c["container_id"])))

            c_type = c.get("container_type") or "LEFT_RIGHT"
            status_item = QTableWidgetItem(f"{c_type} ({c['status']})")
            status_item.setForeground(QColor("#107C41" if c["status"] == "stable" else "#FF8C00"))
            self.container_table.setItem(row, 2, status_item)

            # Container number detection
            c_num_det = c.get("detections", {}).get("container_number")
            iso_det = c.get("detections", {}).get("iso_code")

            num_text = ""
            iso_badge = "-"
            badge_color = "#888"
            conf_val = 0.0

            if c_num_det:
                num_text = c_num_det.get("manual_override_text") or c_num_det.get("ocr_clean_text") or c_num_det.get("ocr_raw_text") or "-"
                conf_val = c_num_det.get("confidence", 0.0)
                if c_num_det.get("is_valid_iso"):
                    iso_badge = "VALID (ISO 6346)"
                    badge_color = "#107C41"
                else:
                    iso_badge = "CHECK DIGIT / FORMAT ERROR"
                    badge_color = "#D83B01"

            self.container_table.setItem(row, 3, QTableWidgetItem(num_text))

            badge_item = QTableWidgetItem(iso_badge)
            badge_item.setForeground(QColor(badge_color))
            self.container_table.setItem(row, 4, badge_item)

            iso_text = iso_det.get("ocr_clean_text") or iso_det.get("ocr_raw_text") or "-" if iso_det else "-"
            self.container_table.setItem(row, 5, QTableWidgetItem(iso_text))
            self.container_table.setItem(row, 6, QTableWidgetItem(f"{conf_val:.2f}" if conf_val > 0 else "-"))

            reviewed = "YES" if (c_num_det and c_num_det.get("is_reviewed")) else "NO"
            rev_item = QTableWidgetItem(reviewed)
            if reviewed == "YES":
                rev_item.setForeground(QColor("#0078D4"))
            self.container_table.setItem(row, 7, rev_item)

            # Store container and detection ids
            self.container_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, c["id"])
            if c_num_det:
                self.container_table.item(row, 3).setData(Qt.ItemDataRole.UserRole, c_num_det["id"])

        if containers:
            selected_row_to_restore = 0
            if self.selected_container_id is not None:
                for r in range(len(containers)):
                    item = self.container_table.item(r, 0)
                    if item and item.data(Qt.ItemDataRole.UserRole) == self.selected_container_id:
                        selected_row_to_restore = r
                        break
            self.container_table.selectRow(selected_row_to_restore)

    def _on_container_selected(self) -> None:
        selected_rows = self.container_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()

        c_id = self.container_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.selected_container_id = c_id

        c_num_item = self.container_table.item(row, 3)
        self.selected_detection_id = c_num_item.data(Qt.ItemDataRole.UserRole) if c_num_item else None
        curr_text = c_num_item.text() if c_num_item else ""
        if curr_text == "-":
            curr_text = ""
        self.edit_manual_num.setText(curr_text)

        # Load image previews
        detections = self.db.get_detections_for_container(c_id)
        containers = [c for c in self.db.get_containers_for_job(self.current_job_id or 0) if c["id"] == c_id]
        if not containers:
            return
        c_info = containers[0]

        # Panorama Image
        pano_path = c_info.get("annotated_image_path") or c_info.get("image_path")
        if pano_path and Path(pano_path).is_file():
            pix = QPixmap(pano_path)
            scaled = pix.scaled(
                self.lbl_pano_img.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_pano_img.setPixmap(scaled)
        else:
            self.lbl_pano_img.setText("Citra panorama sedang diproses oleh Stitching Engine (ALIKED)...")

        # Crops
        self.lbl_crop_num.setText("Crop nomor tidak tersedia")
        self.lbl_crop_iso.setText("Crop ISO code tidak tersedia")

        for d in detections:
            crop_p = d.get("crop_image_path")
            if crop_p and Path(crop_p).is_file():
                pix = QPixmap(crop_p)
                scaled = pix.scaled(
                    QSize(300, 120),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                if d["detection_type"] == "container_number":
                    self.lbl_crop_num.setPixmap(scaled)
                elif d["detection_type"] == "iso_code":
                    self.lbl_crop_iso.setPixmap(scaled)

    def _on_manual_text_changed(self, text: str) -> None:
        res = validate_iso6346(text)
        if not text:
            self.lbl_iso_feedback.setText("-")
            self.lbl_iso_feedback.setStyleSheet("color: #888;")
        elif res.is_valid:
            self.lbl_iso_feedback.setText(" Check Digit Sesuai (ISO 6346)")
            self.lbl_iso_feedback.setStyleSheet("color: #107C41; font-weight: bold;")
        else:
            exp = f"Expected: {res.expected_check_digit}" if res.expected_check_digit is not None else ""
            self.lbl_iso_feedback.setText(f" Tidak Valid: {res.error_message or ''} {exp}")
            self.lbl_iso_feedback.setStyleSheet("color: #D83B01; font-weight: bold;")

    def _on_save_manual_override(self) -> None:
        if not self.selected_detection_id:
            QMessageBox.warning(self, "Peringatan", "Pilih kontainer yang memiliki data nomor terlebih dahulu!")
            return

        text = self.edit_manual_num.text().strip().upper()
        if not text:
            QMessageBox.warning(self, "Peringatan", "Nomor kontainer tidak boleh kosong!")
            return

        self.cmd_queue.put({
            "action": "MANUAL_OVERRIDE",
            "detection_id": self.selected_detection_id,
            "text": text,
        })
        self.show_toast(f"Koreksi disimpan: {text}", "success")

    def _on_play_input_video(self) -> None:
        if not self.selected_video_path or not Path(self.selected_video_path).is_file():
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file video drone terlebih dahulu!")
            return
        if hasattr(self, "_active_video_dlg") and self._active_video_dlg is not None:
            try:
                self._active_video_dlg.close()
            except Exception:
                pass
            self._active_video_dlg = None

        try:
            self._active_video_dlg = VideoPlayerDialog(
                self,
                self.selected_video_path,
                f"Pemutar Video Drone (Input) - {Path(self.selected_video_path).name}",
            )
            self._active_video_dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error Pemutar Video", f"Gagal memutar video:\n{e}\n\nAnda dapat membuka file langsung dengan pemutar default Windows.")
        finally:
            self._active_video_dlg = None

    def _on_play_output_video(self) -> None:
        if not self.current_job_id:
            QMessageBox.information(self, "Informasi", "Silakan pilih job pada tabel riwayat terlebih dahulu.")
            return
        out_vid = self._get_job_output_video(self.current_job_id)
        if not out_vid or not Path(out_vid).is_file():
            QMessageBox.information(
                self,
                "Informasi",
                "Video hasil anotasi belum tersedia untuk job ini.\n"
                "(Video otomatis dirender dan disimpan dalam folder sesi setelah pemrosesan selesai).",
            )
            return
        if hasattr(self, "_active_video_dlg") and self._active_video_dlg is not None:
            try:
                self._active_video_dlg.close()
            except Exception:
                pass
            self._active_video_dlg = None

        try:
            self._active_video_dlg = VideoPlayerDialog(
                self,
                out_vid,
                f"Pemutar Video Hasil Anotasi - Job #{self.current_job_id}",
            )
            self._active_video_dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error Pemutar Video", f"Gagal memutar video:\n{e}\n\nAnda dapat membuka file langsung dengan pemutar default Windows.")
        finally:
            self._active_video_dlg = None

    def _resolve_job_output_dir(self, job_id: int) -> Optional[Path]:
        """Resolves the session output directory for a job across DB output_video_path, images, or folder name."""
        out_dir = self.config.paths.resolved_output_dir()
        jobs = [j for j in self.db.get_all_jobs() if j["id"] == job_id]
        if jobs and jobs[0].get("output_video_path"):
            p = Path(jobs[0]["output_video_path"])
            if p.parent.is_dir():
                return p.parent

        containers = self.db.get_containers_for_job(job_id)
        for c in containers:
            img_p_str = c["image_path"] if isinstance(c, dict) else getattr(c, "image_path", None)
            if img_p_str:
                p = Path(img_p_str).resolve()
                for parent in [p] + list(p.parents):
                    if parent.parent.resolve() == out_dir.resolve():
                        return parent

        if jobs:
            v_stem = Path(jobs[0].get("video_path", "")).stem
            j_name = jobs[0].get("name", "")
            for pat in [f"*_{v_stem}" if v_stem else "", f"*_{j_name}" if j_name else ""]:
                if not pat:
                    continue
                candidates = sorted(out_dir.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
                if candidates:
                    return candidates[0]
        return None

    def _get_job_output_video(self, job_id: int) -> Optional[str]:
        jobs = [j for j in self.db.get_all_jobs() if j["id"] == job_id]
        if jobs and jobs[0].get("output_video_path"):
            p = Path(jobs[0]["output_video_path"])
            if p.is_file():
                return str(p)
        job_dir = self._resolve_job_output_dir(job_id)
        if job_dir:
            vids = list(job_dir.glob("*_annotated.mp4"))
            if vids:
                return str(vids[0])
        return None

    def _on_open_output_folder(self) -> None:
        """Opens the output directory of the selected or active job in Windows File Explorer."""
        out_dir = self.config.paths.resolved_output_dir()
        target_dir = out_dir

        try:
            if self.current_job_id:
                resolved = self._resolve_job_output_dir(self.current_job_id)
                if resolved:
                    target_dir = resolved
        except Exception as e:
            print(f"[WARN] Error resolving job output directory: {e}")

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            import os
            os.startfile(str(target_dir))
        except Exception:
            try:
                import subprocess
                subprocess.Popen(["explorer", str(target_dir)])
            except Exception as e2:
                QMessageBox.warning(self, "Buka Folder", f"Gagal membuka folder:\n{target_dir}\n\nError: {e2}")

    def _on_export_csv(self) -> None:
        """Exports the container table of the selected job to a CSV report."""
        if not self.current_job_id:
            QMessageBox.information(self, "Ekspor CSV", "Silakan pilih job pada tabel riwayat terlebih dahulu.")
            return
        containers = self.db.get_full_container_view(self.current_job_id)
        if not containers:
            QMessageBox.information(self, "Ekspor CSV", "Tidak ada data kontainer pada job ini.")
            return

        default_name = f"hasil_ocr_job_{self.current_job_id}.csv"
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan CSV", default_name, "CSV Files (*.csv)")
        if not file_path:
            return

        import csv
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Region",
                    "Container ID",
                    "Mode / Status",
                    "Nomor Kontainer (OCR)",
                    "ISO 6346 Check",
                    "ISO Code",
                    "Confidence",
                    "Manual Review",
                ])
                for c in containers:
                    c_num = c.get("detections", {}).get("container_number", {})
                    iso_d = c.get("detections", {}).get("iso_code", {})
                    num_text = c_num.get("manual_override_text") or c_num.get("ocr_clean_text") or "-"
                    iso_check = "VALID" if c_num.get("is_valid_iso") else "INVALID"
                    iso_code = iso_d.get("ocr_clean_text") or "-"
                    conf = f"{c_num.get('confidence', 0.0):.2f}"
                    rev = "YES" if c_num.get("is_reviewed") else "NO"
                    writer.writerow([
                        f"R{c['route_order']:03d}",
                        c["container_id"],
                        c.get("container_type", "FRONT_REAR"),
                        num_text,
                        iso_check,
                        iso_code,
                        conf,
                        rev,
                    ])
            QMessageBox.information(self, "Sukses", f"Data berhasil diekspor ke:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Gagal Ekspor", f"Terjadi kesalahan saat menyimpan CSV:\n{e}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.cmd_queue.put({"action": "UPDATE_CONFIG", "config": self.config.to_dict()})
            dev_lbl = str(getattr(self.config.stitching, "device", "cpu")).upper()
            self.show_toast(f"Pengaturan berhasil disimpan (Mode Komputasi: {dev_lbl}).", "success")

    def _on_logout(self) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Logout",
            "Apakah Anda yakin ingin keluar dari sesi akun ini?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from src.ui.login_dialog import LoginDialog
            from src.utils.security import store_auth_token

            store_auth_token("")
            self.hide()
            dlg = LoginDialog(self.config, self)
            if dlg.exec() == LoginDialog.DialogCode.Accepted:
                self.show()
                self.show_toast("Berhasil masuk dengan akun baru.", "info")
            else:
                self.close()

    def show_toast(self, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        self._toasts.show(message, level, duration_ms)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_toasts", None) is not None:
            self._toasts.reposition()

    def closeEvent(self, event) -> None:
        """Handles application shutdown cleanly according to TDD v1.9."""
        if self.is_job_running:
            reply = QMessageBox.question(
                self,
                "Konfirmasi Keluar",
                "Job pemrosesan sedang berlangsung!\n\nApakah Anda ingin membatalkan job aktif dan keluar?\n(Job yang terhenti dapat di-resume kembali saat aplikasi dibuka)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.orchestrator:
                self.orchestrator.cancel_current_job()

        # 0. Cleanly close any open video player dialog to free WMF / DirectX devices
        if hasattr(self, "_active_video_dlg") and self._active_video_dlg is not None:
            try:
                self._active_video_dlg.close()
            except Exception:
                pass
            self._active_video_dlg = None

        # 1. Stop polling timer immediately so event loop isn't kept alive
        if hasattr(self, "event_timer") and self.event_timer.isActive():
            self.event_timer.stop()

        # 2. Signal Orchestrator to stop
        if hasattr(self, "cmd_queue") and self.cmd_queue is not None:
            try:
                self.cmd_queue.put({"action": "STOP"})
            except Exception:
                pass

        # 3. Accept close event
        event.accept()

        # 4. Explicitly quit QApplication so app.exec() terminates immediately
        app_inst = QApplication.instance()
        if app_inst:
            app_inst.quit()
