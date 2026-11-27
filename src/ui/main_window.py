"""
Main Application Window (PySide6).
"""
from __future__ import annotations

import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from src.config import BASE_DIR, AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator, OrchestratorEvent
from src.recognition.detector import describe_compute_plan
from src.ui.settings_dialog import SettingsDialog
from src.ui.toast import ElidedLabel, ToastManager
from src.ui.video_player_dialog import VideoPlayerDialog
from src.utils.iso6346 import validate_iso6346, validate_iso_size_type


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
        self.selected_iso_detection_id: Optional[int] = None
        self.is_job_running = False

        self._orig_crop_num_pixmap: Optional[QPixmap] = None
        self._orig_crop_iso_pixmap: Optional[QPixmap] = None

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

        # Left: Job History & Log
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

        # Right: Container Table & Preview (Using Vertical Splitter)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_vertical_splitter = QSplitter(Qt.Orientation.Vertical)

        # Table Header
        right_header_lay = QHBoxLayout()
        right_header_lay.addWidget(QLabel("<b>Hasil Deteksi & Pengenalan Kontainer (ISO 6346):</b>"))
        self.lbl_container_summary = QLabel("")
        self.lbl_container_summary.setStyleSheet("color: #888; font-size: 11px; margin-left: 8px;")
        right_header_lay.addWidget(self.lbl_container_summary)
        right_header_lay.addStretch()

        self.btn_play_output = QPushButton("Putar Video Hasil")
        self.btn_play_output.setStyleSheet("padding: 3px 8px; font-size: 11px; font-weight: bold; background-color: #005a9e; color: white;")
        self.btn_play_output.setEnabled(False)
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
        self.container_table.setColumnWidth(4, 160)
        self.container_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.container_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.container_table.itemSelectionChanged.connect(self._on_container_selected)
        
        right_vertical_splitter.addWidget(self.container_table)

        # Bottom Inspection View
        preview_group = QGroupBox("Detail Visual & Verifikasi Lapangan (Zero Double-Detection)")
        preview_layout = QVBoxLayout(preview_group)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panorama
        pano_box = QWidget()
        pano_lay = QVBoxLayout(pano_box)
        pano_lay.setContentsMargins(0, 0, 0, 0)
        pano_lay.addWidget(QLabel("Citra Kontainer Terdeteksi (RAW / Anotasi Bounding Box):"))
        self.lbl_pano_img = QLabel("Pilih baris kontainer untuk menampilkan citra")
        self.lbl_pano_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_pano_img.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444;")
        self.lbl_pano_img.setMinimumHeight(150)
        pano_lay.addWidget(self.lbl_pano_img)
        preview_splitter.addWidget(pano_box)

        # Crops & Re-OCR
        crop_box = QWidget()
        crop_lay = QVBoxLayout(crop_box)
        crop_lay.setContentsMargins(0, 0, 0, 0)
        crop_lay.addWidget(QLabel("Crop Resolusi Penuh (Dikirim ke Halotec OCR API):"))

        crop_tabs = QTabWidget()

        # Tab Number
        tab_num = QWidget()
        tab_num_lay = QVBoxLayout(tab_num)
        self.lbl_crop_num = QLabel("Crop Nomor Kontainer")
        self.lbl_crop_num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_num.setStyleSheet("background-color: #252526; color: #888;")
        tab_num_lay.addWidget(self.lbl_crop_num)
        
        num_rot_lay = QHBoxLayout()
        num_rot_lay.addStretch()
        self.btn_rot_num_ccw = QPushButton("⟲ Putar Kiri")
        self.btn_rot_num_cw = QPushButton("⟳ Putar Kanan")
        self.btn_reocr_num = QPushButton("🔄 OCR Ulang")
        self.btn_reocr_num.setStyleSheet("background-color: #D83B01; color: white; font-weight: bold;")
        
        self.btn_rot_num_ccw.clicked.connect(lambda: self._rotate_crop("num", -90))
        self.btn_rot_num_cw.clicked.connect(lambda: self._rotate_crop("num", 90))
        self.btn_reocr_num.clicked.connect(lambda: self._on_re_ocr_crop("num"))
        
        num_rot_lay.addWidget(self.btn_rot_num_ccw)
        num_rot_lay.addWidget(self.btn_rot_num_cw)
        num_rot_lay.addWidget(self.btn_reocr_num)
        num_rot_lay.addStretch()
        tab_num_lay.addLayout(num_rot_lay)
        crop_tabs.addTab(tab_num, "Nomor Kontainer")

        # Tab ISO
        tab_iso = QWidget()
        tab_iso_lay = QVBoxLayout(tab_iso)
        self.lbl_crop_iso = QLabel("Crop ISO Code")
        self.lbl_crop_iso.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_iso.setStyleSheet("background-color: #252526; color: #888;")
        tab_iso_lay.addWidget(self.lbl_crop_iso)
        
        iso_rot_lay = QHBoxLayout()
        iso_rot_lay.addStretch()
        self.btn_rot_iso_ccw = QPushButton("⟲ Putar Kiri")
        self.btn_rot_iso_cw = QPushButton("⟳ Putar Kanan")
        self.btn_reocr_iso = QPushButton("🔄 OCR Ulang")
        self.btn_reocr_iso.setStyleSheet("background-color: #D83B01; color: white; font-weight: bold;")
        
        self.btn_rot_iso_ccw.clicked.connect(lambda: self._rotate_crop("iso", -90))
        self.btn_rot_iso_cw.clicked.connect(lambda: self._rotate_crop("iso", 90))
        self.btn_reocr_iso.clicked.connect(lambda: self._on_re_ocr_crop("iso"))
        
        iso_rot_lay.addWidget(self.btn_rot_iso_ccw)
        iso_rot_lay.addWidget(self.btn_rot_iso_cw)
        iso_rot_lay.addWidget(self.btn_reocr_iso)
        iso_rot_lay.addStretch()
        tab_iso_lay.addLayout(iso_rot_lay)
        crop_tabs.addTab(tab_iso, "ISO Code")

        crop_lay.addWidget(crop_tabs)
        preview_splitter.addWidget(crop_box)
        preview_layout.addWidget(preview_splitter)

        # Manual Review Bar (Number)
        review_bar = QHBoxLayout()
        review_bar.addWidget(QLabel("<b>Koreksi Manual (Nomor):</b>"))
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

        # Manual Review Bar (ISO)
        iso_review_bar = QHBoxLayout()
        iso_review_bar.addWidget(QLabel("<b>Koreksi ISO Code:</b>"))
        self.edit_manual_iso = QLineEdit()
        self.edit_manual_iso.setPlaceholderText("Edit ISO (mis. 45G1)...")
        self.edit_manual_iso.textChanged.connect(self._on_manual_iso_text_changed)
        iso_review_bar.addWidget(self.edit_manual_iso)

        self.lbl_iso_feedback_iso = QLabel("-")
        iso_review_bar.addWidget(self.lbl_iso_feedback_iso)

        self.btn_save_iso = QPushButton("Simpan ISO")
        self.btn_save_iso.setStyleSheet("background-color: #107C41; color: white; font-weight: bold; padding: 4px 12px;")
        self.btn_save_iso.clicked.connect(self._on_save_manual_iso)
        iso_review_bar.addWidget(self.btn_save_iso)
        preview_layout.addLayout(iso_review_bar)

        right_vertical_splitter.addWidget(preview_group)
        right_vertical_splitter.setStretchFactor(0, 3)
        right_vertical_splitter.setStretchFactor(1, 2)

        right_layout.addWidget(right_vertical_splitter)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 7)
        main_layout.addWidget(main_splitter)

        self.selected_video_path: Optional[str] = None

    def _init_event_timer(self) -> None:
        self.event_timer = QTimer(self)
        self.event_timer.setInterval(100)
        self.event_timer.timeout.connect(self._process_orchestrator_events)
        self.event_timer.start()

    def _scroll_activity_log_to_bottom(self) -> None:
        scroll_bar = self.txt_activity_log.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _append_activity_log(self, message: str) -> None:
        self.txt_activity_log.appendPlainText(message)
        QTimer.singleShot(0, self._scroll_activity_log_to_bottom)

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
                self._load_jobs()
            elif evt.type == "PROGRESS":
                pct = int(evt.data.get("progress", 0.0) * 100)
                self.progress_bar.setValue(pct)
                msg = evt.data.get("message", "Memproses...")
                self.lbl_status.setText(msg)
                log_entry = evt.data.get("log")
                if log_entry:
                    self.txt_activity_log.appendPlainText(f"[{now_str}] {log_entry}")
            elif evt.type == "CONTAINER_PROCESSED":
                if self.current_job_id == evt.job_id:
                    self._load_containers_for_job(evt.job_id)
                    self.btn_reocr_num.setEnabled(True)
                    self.btn_reocr_num.setText("🔄 OCR Ulang")
                    self.btn_reocr_iso.setEnabled(True)
                    self.btn_reocr_iso.setText("🔄 OCR Ulang")

                    message = str(evt.data.get("message", "")).strip()
                    if message.startswith("Re-OCR sukses"):
                        self.show_toast(message, "success", 6000)
            elif evt.type == "JOB_COMPLETED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(100)
                self.show_toast("Job Selesai.", "success")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [SUKSES] {evt.data.get('message', 'Job selesai!')}\n")
                self._load_jobs()
                if self.current_job_id == evt.job_id:
                    self._load_containers_for_job(evt.job_id)
            elif evt.type == "JOB_FAILED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(0)
                self.show_toast("Job Gagal.", "error")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [GAGAL] {evt.data.get('error')}\n")
                self._load_jobs()
                QMessageBox.critical(self, "Error Job", f"Job {evt.job_id} gagal:\n{evt.data.get('error')}")
            elif evt.type == "JOB_CANCELLED":
                self._set_start_button_state(False)
                self.progress_bar.setValue(0)
                self.show_toast("Proses Dibatalkan.", "warning")
                self.txt_activity_log.appendPlainText(f"[{now_str}] [DIBATALKAN] {evt.data.get('message', 'Proses dibatalkan.')}\n")
                self._load_jobs()
            elif evt.type == "MANUAL_OVERRIDE_SAVED":
                self.btn_save_override.setText("Simpan Koreksi")
                self.btn_save_override.setEnabled(True)
                saved_text = evt.data.get("text", "")
                self.show_toast(
                    f"Koreksi nomor container berhasil disimpan{': ' + saved_text if saved_text else ''}",
                    "success", 6000
                )
                if evt.job_id == self.current_job_id:
                    self._load_containers_for_job(self.current_job_id)
            elif evt.type == "MANUAL_ISO_OVERRIDE_SAVED":
                # Pulihkan tombol tanpa bergantung pada job yang sedang dipilih.
                self.btn_save_iso.setEnabled(True)
                self.btn_save_iso.setText("Simpan ISO")
                saved_text = str(evt.data.get("text", "")).strip()
                self._append_activity_log(
                    f"[{now_str}] [MANUAL ISO SUKSES] container_id={evt.data.get('container_id')} "
                    f"ISO={saved_text} tersimpan pada job #{evt.job_id}"
                )
                self.show_toast(f"Koreksi ISO berhasil disimpan: {saved_text}", "success", 6000)
                if self.current_job_id == evt.job_id:
                    self._load_containers_for_job(evt.job_id)
            elif evt.type == "MANUAL_ISO_OVERRIDE_FAILED":
                self.btn_save_iso.setEnabled(True)
                self.btn_save_iso.setText("Simpan ISO")
                error_text = str(evt.data.get("error", "Penyimpanan ISO gagal"))
                self._append_activity_log(
                    f"[{now_str}] [MANUAL ISO GAGAL] container_id={evt.data.get('container_id')}: {error_text}"
                )
                self.show_toast(f"Koreksi ISO gagal: {error_text}", "error", 8000)
            elif evt.type == "LOG":
                log_entry = evt.data.get("log") or evt.data.get("message")
                if log_entry:
                    if not log_entry.startswith("["):
                        log_entry = f"[{now_str}] {log_entry}"
                    self.txt_activity_log.appendPlainText(log_entry)

                    # Pulihkan tombol dan beri feedback jika request manual gagal.
                    lower_log = log_entry.lower()
                    if "re-ocr error" in lower_log or "reocr error" in lower_log:
                        self.btn_reocr_num.setEnabled(True)
                        self.btn_reocr_num.setText("🔄 OCR Ulang")
                        self.btn_reocr_iso.setEnabled(True)
                        self.btn_reocr_iso.setText("🔄 OCR Ulang")
                        self.show_toast("OCR ulang gagal. Periksa log aktivitas dan token/API.", "error", 7000)
                    elif "manual iso" in lower_log and ("error" in lower_log or "ditolak" in lower_log):
                        self.btn_save_iso.setEnabled(True)
                        self.btn_save_iso.setText("Simpan ISO")
                        self.show_toast("Koreksi ISO gagal disimpan. Periksa format atau log aktivitas.", "error", 7000)
                    elif "manual override" in lower_log and ("error" in lower_log or "ditolak" in lower_log):
                        self.btn_save_override.setEnabled(True)
                        self.btn_save_override.setText("Simpan Koreksi")
                        self.show_toast("Koreksi nomor container gagal disimpan. Periksa log aktivitas.", "error", 7000)

    def _on_select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pilih File Video Drone", str(BASE_DIR), "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)")
        if path:
            self.selected_video_path = path
            self.btn_new_job.setText(f"📁 {Path(path).name}")
            self.show_toast(f"Video dipilih: {Path(path).name}", "info")
            self.btn_play_input.setEnabled(True)

    def _on_mode_changed(self) -> None:
        self._update_cam_pos_options(self.mode_combo.currentData())

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
            self.btn_start.setStyleSheet("background-color: #D83B01; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        else:
            self.btn_start.setText("Mulai Proses")
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet("background-color: #0078D4; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")

    def _on_start_or_cancel_job(self) -> None:
        if self.is_job_running:
            reply = QMessageBox.question(self, "Konfirmasi", "Batalkan proses yang berjalan?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.btn_start.setEnabled(False)
                self.btn_start.setText("Membatalkan...")
                self.lbl_status.setText("Membatalkan proses...")
                if self.orchestrator: self.orchestrator.cancel_current_job()
                self.cmd_queue.put({"action": "CANCEL_JOB"})
        else:
            self._on_start_job()

    def _clear_workspace_for_new_job(self) -> None:
        self.container_table.setRowCount(0)
        self.txt_activity_log.clear()
        self.lbl_pano_img.setText("Memproses video...")
        self.lbl_pano_img.setPixmap(QPixmap())
        self.lbl_crop_num.setText("Menunggu deteksi...")
        self.lbl_crop_num.setPixmap(QPixmap())
        self.lbl_crop_iso.setText("Menunggu deteksi...")
        self.lbl_crop_iso.setPixmap(QPixmap())
        self._orig_crop_num_pixmap = None
        self._orig_crop_iso_pixmap = None
        self.edit_manual_num.clear()
        self.edit_manual_iso.clear()
        self.lbl_iso_feedback.setText("-")
        self.lbl_iso_feedback_iso.setText("-")
        self.btn_save_override.setEnabled(False)
        self.btn_save_iso.setEnabled(False)
        self.btn_reocr_num.setEnabled(False)
        self.btn_reocr_iso.setEnabled(False)
        self.selected_container_id = None
        self.selected_detection_id = None
        self.selected_iso_detection_id = None
        self.lbl_container_summary.setText("Memulai proses baru...")
        self.btn_play_output.setEnabled(False)

    def _on_start_job(self) -> None:
        if not self.selected_video_path:
            QMessageBox.warning(self, "Peringatan", "Pilih video terlebih dahulu!")
            return

        self._clear_workspace_for_new_job()
        self.cmd_queue.put({
            "action": "START_JOB",
            "name": Path(self.selected_video_path).stem,
            "video_path": self.selected_video_path,
            "mode": self.mode_combo.currentData(),
            "camera_position": self.cam_pos_combo.currentData() or "left",
        })

    def _load_jobs(self) -> None:
        """Refresh job history newest-first and keep the active/current job selected."""
        jobs = self.db.get_all_jobs() or []
        jobs = sorted(jobs, key=lambda item: int(item.get("id", 0)), reverse=True)

        was_blocked = self.jobs_table.blockSignals(True)
        target_row = None
        try:
            self.jobs_table.setRowCount(len(jobs))

            for row, job in enumerate(jobs):
                job_id = int(job.get("id", 0))
                self.jobs_table.setItem(row, 0, QTableWidgetItem(str(job_id)))
                self.jobs_table.setItem(row, 1, QTableWidgetItem(str(job.get("name") or "")))
                self.jobs_table.setItem(row, 2, QTableWidgetItem(str(job.get("mode") or "")))

                status = str(job.get("status") or "")
                status_item = QTableWidgetItem(status)
                if status == "COMPLETED":
                    status_item.setForeground(QColor("#107C41"))
                elif status == "PROCESSING":
                    status_item.setForeground(QColor("#0078D4"))
                elif status == "FAILED":
                    status_item.setForeground(QColor("#D83B01"))
                elif status == "INTERRUPTED":
                    status_item.setForeground(QColor("#FF8C00"))
                self.jobs_table.setItem(row, 3, status_item)

                if self.current_job_id is not None and job_id == int(self.current_job_id):
                    target_row = row

            # If there is no current job, select the newest one by default.
            if jobs and target_row is None:
                target_row = 0
                self.current_job_id = int(jobs[0].get("id", 0))

            self.jobs_table.clearSelection()
            if target_row is not None:
                self.jobs_table.selectRow(target_row)
                item = self.jobs_table.item(target_row, 0)
                if item is not None:
                    self.jobs_table.scrollToItem(
                        item, QAbstractItemView.ScrollHint.PositionAtTop
                    )
        finally:
            self.jobs_table.blockSignals(was_blocked)

        # Signals were blocked while rebuilding; load the selected job explicitly.
        if target_row is not None and jobs:
            selected_job_id = int(jobs[target_row].get("id", 0))
            self.current_job_id = selected_job_id
            self._load_containers_for_job(selected_job_id)

    def _on_job_selected(self) -> None:
        selected_rows = self.jobs_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        job_id_item = self.jobs_table.item(row, 0)
        if not job_id_item:
            return

        job_id = int(job_id_item.text())
        if job_id != self.current_job_id:
            # Clear stale selection IDs from the previously selected job.
            self.selected_container_id = None
            self.selected_detection_id = None
            self.selected_iso_detection_id = None
            self.edit_manual_num.clear()
            self.edit_manual_iso.clear()
            self.btn_save_override.setEnabled(False)
            self.btn_save_iso.setEnabled(False)
            self._orig_crop_num_pixmap = None
            self._orig_crop_iso_pixmap = None
            self.lbl_pano_img.setPixmap(QPixmap())
            self.lbl_pano_img.setText("Pilih baris kontainer untuk menampilkan citra")
            self.lbl_crop_num.setPixmap(QPixmap())
            self.lbl_crop_num.setText("Crop nomor tidak tersedia")
            self.lbl_crop_iso.setPixmap(QPixmap())
            self.lbl_crop_iso.setText("Crop ISO code tidak tersedia")

        self.current_job_id = job_id
        self._load_containers_for_job(job_id)

    @staticmethod
    def _resolve_saved_path(raw_path: Optional[str]) -> Optional[Path]:
        """Resolve persisted paths whether they were stored absolute or relative."""
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        candidates = [path] if path.is_absolute() else [BASE_DIR / path, Path.cwd() / path]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except OSError:
                continue
        return None

    def _load_containers_for_job(self, job_id: int) -> None:
        self.btn_play_output.setEnabled(bool(self._get_job_output_video(job_id)))
        containers = self.db.get_full_container_view(job_id)
        self.container_table.setRowCount(len(containers))

        total_cnt = len(containers)
        valid_cnt = sum(1 for c in containers if c.get("detections", {}).get("container_number", {}).get("is_valid_iso"))
        
        if total_cnt > 0:
            self.lbl_container_summary.setText(f"Total: {total_cnt} | Valid ISO: {valid_cnt} | Perlu Review: {total_cnt - valid_cnt}")
            self.lbl_container_summary.setStyleSheet("color: #107C41; font-weight: bold; background-color: #1b3323; border: 1px solid #107C41; border-radius: 4px; padding: 2px 8px; font-size: 11px;")
        else:
            self.lbl_container_summary.setText("Belum ada data")
            self.lbl_container_summary.setStyleSheet("color: #888;")

        for row, c in enumerate(containers):
            self.container_table.setItem(row, 0, QTableWidgetItem(f"R{c['route_order']:03d}"))
            self.container_table.setItem(row, 1, QTableWidgetItem(str(c["container_id"])))
            
            c_type = c.get("container_type") or "LEFT_RIGHT"
            status_item = QTableWidgetItem(f"{c_type} ({c['status']})")
            status_item.setForeground(QColor("#107C41" if c["status"] == "stable" else "#FF8C00"))
            self.container_table.setItem(row, 2, status_item)

            c_num_det = c.get("detections", {}).get("container_number")
            iso_det = c.get("detections", {}).get("iso_code")

            num_text = ""
            iso_badge = "-"
            badge_color = "#888"
            conf_val = 0.0

            if c_num_det:
                is_manual = bool(c_num_det.get("is_reviewed"))
                if is_manual and c_num_det.get("manual_override_text"):
                    num_text = c_num_det.get("manual_override_text")
                else:
                    num_text = c_num_det.get("ocr_clean_text") or c_num_det.get("ocr_raw_text") or "-"
                conf_val = c_num_det.get("confidence", 0.0)
                
                if c_num_det.get("is_valid_iso"):
                    if is_manual:
                        iso_badge = "VALID (Manual Fix)"
                        badge_color = "#0078D4"
                    else:
                        iso_badge = "VALID (Auto OCR)"
                        badge_color = "#107C41"
                else:
                    iso_badge = "CHECK DIGIT / FORMAT ERROR"
                    badge_color = "#D83B01"

            num_item = QTableWidgetItem(num_text)
            if c_num_det and c_num_det.get("is_reviewed"):
                num_item.setForeground(QColor("#4fc1ff"))
            self.container_table.setItem(row, 3, num_item)

            badge_item = QTableWidgetItem(iso_badge)
            badge_item.setForeground(QColor(badge_color))
            badge_item.setToolTip(iso_badge)
            self.container_table.setItem(row, 4, badge_item)

            iso_text = "-"
            if iso_det:
                iso_text = iso_det.get("manual_override_text") or iso_det.get("ocr_clean_text") or iso_det.get("ocr_raw_text") or "-"
            
            # Jangan tampilkan "-" jika isinya kosong string
            if iso_text.strip() == "": iso_text = "-"
            
            self.container_table.setItem(row, 5, QTableWidgetItem(iso_text))
            self.container_table.setItem(row, 6, QTableWidgetItem(f"{conf_val:.2f}" if conf_val > 0 else "-"))

            # Manual Review harus mencakup koreksi pada nomor container maupun ISO Code.
            # ISO Code disimpan sebagai detection terpisah, jadi jangan hanya membaca
            # flag is_reviewed dari c_num_det.
            num_reviewed = bool(c_num_det and c_num_det.get("is_reviewed"))
            iso_reviewed = bool(iso_det and iso_det.get("is_reviewed"))
            is_reviewed = num_reviewed or iso_reviewed
            reviewed = "YES" if is_reviewed else "NO"
            rev_item = QTableWidgetItem(reviewed)
            if is_reviewed:
                rev_item.setForeground(QColor("#0078D4"))
                review_sources = []
                if num_reviewed:
                    review_sources.append("Nomor Container")
                if iso_reviewed:
                    review_sources.append("ISO Code")
                rev_item.setToolTip("Koreksi manual tersimpan: " + ", ".join(review_sources))
            else:
                rev_item.setToolTip("Belum ada koreksi manual tersimpan")
            self.container_table.setItem(row, 7, rev_item)

            self.container_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, c["id"])
            if c_num_det:
                self.container_table.item(row, 3).setData(Qt.ItemDataRole.UserRole, c_num_det["id"])

        if containers:
            selected_row = 0
            if self.selected_container_id is not None:
                for r in range(len(containers)):
                    item = self.container_table.item(r, 0)
                    if item and item.data(Qt.ItemDataRole.UserRole) == self.selected_container_id:
                        selected_row = r
                        break
            self.container_table.clearSelection()
            self.container_table.selectRow(selected_row)
            # Explicitly load the selected row even if Qt emits no selection signal.
            self._on_container_selected()
        else:
            self.selected_container_id = None
            self.selected_detection_id = None
            self.selected_iso_detection_id = None
            self.btn_save_override.setEnabled(False)
            self.btn_save_iso.setEnabled(False)
            self.lbl_pano_img.setPixmap(QPixmap())
            self.lbl_pano_img.setText("Job ini belum memiliki data kontainer.")
            self.lbl_crop_num.setPixmap(QPixmap())
            self.lbl_crop_num.setText("Crop nomor tidak tersedia")
            self.lbl_crop_iso.setPixmap(QPixmap())
            self.lbl_crop_iso.setText("Crop ISO code tidak tersedia")

    def _on_container_selected(self) -> None:
        selected_rows = self.container_table.selectionModel().selectedRows()
        if not selected_rows: return
        row = selected_rows[0].row()

        c_id = self.container_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.selected_container_id = c_id

        c_num_item = self.container_table.item(row, 3)
        self.selected_detection_id = c_num_item.data(Qt.ItemDataRole.UserRole) if c_num_item else None
        
        detections = self.db.get_detections_for_container(c_id)
        
        # Cari ID deteksi ISO
        self.selected_iso_detection_id = None
        iso_det = next((d for d in detections if d["detection_type"] == "iso_code"), None)
        if iso_det:
            self.selected_iso_detection_id = iso_det["id"]
        
        self.btn_save_override.setEnabled(bool(self.selected_detection_id))
        self.btn_save_iso.setEnabled(True) # Selalu izinkan koreksi ISO jika kontainer dipilih
        
        self.edit_manual_num.setText(c_num_item.text() if c_num_item and c_num_item.text() != "-" else "")
        
        iso_text = "-"
        if iso_det:
            iso_text = iso_det.get("manual_override_text") or iso_det.get("ocr_clean_text") or iso_det.get("ocr_raw_text") or "-"
            if iso_text.strip() == "": iso_text = "-"
        self.edit_manual_iso.setText(iso_text if iso_text != "-" else "")

        containers = [c for c in self.db.get_containers_for_job(self.current_job_id or 0) if c["id"] == c_id]
        if not containers: return
        c_info = containers[0]

        pano_path = self._resolve_saved_path(
            c_info.get("annotated_image_path") or c_info.get("image_path")
        )
        if pano_path:
            pix = QPixmap(str(pano_path))
            scaled = pix.scaled(self.lbl_pano_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_pano_img.setPixmap(scaled)
        else:
            self.lbl_pano_img.setPixmap(QPixmap())
            self.lbl_pano_img.setText("Citra panorama tidak ditemukan.")

        self.lbl_crop_num.setText("Crop nomor tidak tersedia")
        self.lbl_crop_num.setPixmap(QPixmap())
        self.lbl_crop_iso.setText("Crop ISO code tidak tersedia")
        self.lbl_crop_iso.setPixmap(QPixmap())
        self._orig_crop_num_pixmap = None
        self._orig_crop_iso_pixmap = None

        for d in detections:
            crop_p = self._resolve_saved_path(d.get("crop_image_path"))
            if crop_p:
                pix = QPixmap(str(crop_p))
                if not pix.isNull() and d["detection_type"] == "container_number":
                    self._orig_crop_num_pixmap = pix
                    self._render_crop_image("num")
                elif not pix.isNull() and d["detection_type"] == "iso_code":
                    self._orig_crop_iso_pixmap = pix
                    self._render_crop_image("iso")

        # Restore Re-OCR availability from saved crop files when reopening a job.
        self.btn_reocr_num.setEnabled(
            bool(self.selected_detection_id and self._orig_crop_num_pixmap and not self._orig_crop_num_pixmap.isNull())
        )
        self.btn_reocr_iso.setEnabled(
            bool(self.selected_iso_detection_id and self._orig_crop_iso_pixmap and not self._orig_crop_iso_pixmap.isNull())
        )

    def _render_crop_image(self, target: str):
        pixmap = self._orig_crop_num_pixmap if target == "num" else self._orig_crop_iso_pixmap
        label = self.lbl_crop_num if target == "num" else self.lbl_crop_iso
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(QSize(220, 90), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
        else:
            label.setText("Crop tidak tersedia")

    def _rotate_crop(self, target: str, angle: int):
        if target == "num" and self._orig_crop_num_pixmap:
            self._orig_crop_num_pixmap = self._orig_crop_num_pixmap.transformed(QTransform().rotate(angle))
            self._render_crop_image("num")
        elif target == "iso" and self._orig_crop_iso_pixmap:
            self._orig_crop_iso_pixmap = self._orig_crop_iso_pixmap.transformed(QTransform().rotate(angle))
            self._render_crop_image("iso")

    def _on_re_ocr_crop(self, target: str) -> None:
        if target == "num":
            det_id = self.selected_detection_id
            det_type = "container_number"
            pixmap = self._orig_crop_num_pixmap
        else:
            det_id = self.selected_iso_detection_id
            det_type = "iso_code"
            pixmap = self._orig_crop_iso_pixmap

        if not det_id:
            QMessageBox.warning(self, "Peringatan", f"Data deteksi {det_type} belum ada di database. Gunakan koreksi manual.")
            return
        if not pixmap or pixmap.isNull():
            QMessageBox.warning(self, "Peringatan", "Gambar crop tidak tersedia untuk di-OCR ulang!")
            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi OCR Ulang",
            f"Jalankan OCR ulang untuk {det_type.replace('_', ' ')}?\n\n"
            "Hasil OCR terbaru akan menggantikan hasil OCR sebelumnya. Lanjutkan?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        temp_dir = Path(BASE_DIR) / "data" / "temp_crops"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"reocr_{det_type}_{det_id}.png"
        
        if not pixmap.save(str(temp_file), "PNG"):
            QMessageBox.critical(self, "Error", "Gagal menyimpan gambar sementara!")
            return

        btn = self.btn_reocr_num if target == "num" else self.btn_reocr_iso
        btn.setEnabled(False)
        btn.setText("⏳ Memproses...")

        self.cmd_queue.put({
            "action": "RE_OCR_CROP",
            "detection_id": det_id,
            "job_id": self.current_job_id,
            "image_path": str(temp_file),
            "detection_type": det_type,
        })
        self.show_toast(f"Mengirim crop {det_type} ke OCR...", "info")

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
        confirm = QMessageBox.question(
            self,
            "Konfirmasi Koreksi Manual",
            f"Simpan nomor container berikut sebagai koreksi manual?\n\n{text}\n\n"
            "Status Manual Review akan menjadi YES.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.btn_save_override.setEnabled(False)
        self.btn_save_override.setText("Menyimpan...")
        self.cmd_queue.put({
            "action": "MANUAL_OVERRIDE",
            "detection_id": self.selected_detection_id,
            "text": text,
            "job_id": self.current_job_id,
        })

    def _on_manual_iso_text_changed(self, text: str) -> None:
        if not text:
            self.lbl_iso_feedback_iso.setText("-")
            self.lbl_iso_feedback_iso.setStyleSheet("color: #888;")
            return
        is_valid, clean_code, err_msg = validate_iso_size_type(text)
        if is_valid:
            self.lbl_iso_feedback_iso.setText(" Format Sesuai (ISO 6346)")
            self.lbl_iso_feedback_iso.setStyleSheet("color: #107C41; font-weight: bold;")
        else:
            self.lbl_iso_feedback_iso.setText(f" Tidak Valid: {err_msg}")
            self.lbl_iso_feedback_iso.setStyleSheet("color: #D83B01; font-weight: bold;")

    def _on_save_manual_iso(self) -> None:
        if not self.selected_container_id:
            QMessageBox.warning(self, "Peringatan", "Klik baris kontainer di tabel terlebih dahulu!")
            return

        text = self.edit_manual_iso.text().strip().upper()
        is_valid, clean_code, err_msg = validate_iso_size_type(text)
        
        if not is_valid:
            QMessageBox.warning(self, "Format Salah", f"ISO Code tidak valid:\n{err_msg}")
            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Koreksi ISO",
            f"Simpan ISO Code berikut sebagai koreksi manual?\n\n{clean_code}\n\n"
            "Status Manual Review akan menjadi YES.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        target_job_id = self.current_job_id
        target_container_id = self.selected_container_id
        self.btn_save_iso.setEnabled(False)
        self.btn_save_iso.setText("Menyimpan...")
        self._append_activity_log(
            f"[{datetime.now().strftime('%H:%M:%S')}] [MANUAL ISO] Meminta simpan "
            f"container_id={target_container_id}, job=#{target_job_id}, ISO={clean_code}"
        )

        self.cmd_queue.put({
            "action": "MANUAL_ISO_OVERRIDE",
            "container_id": target_container_id,
            "text": clean_code,
            "job_id": target_job_id,
        })
        self.show_toast(f"Menyimpan koreksi ISO: {clean_code}...", "info")

    def _on_play_input_video(self) -> None:
        if not self.selected_video_path or not Path(self.selected_video_path).is_file():
            QMessageBox.warning(self, "Peringatan", "Pilih video drone terlebih dahulu!")
            return
        dlg = VideoPlayerDialog(self, self.selected_video_path, f"Input - {Path(self.selected_video_path).name}")
        dlg.exec()

    def _on_play_output_video(self) -> None:
        if not self.current_job_id: return
        out_vid = self._get_job_output_video(self.current_job_id)
        if not out_vid:
            QMessageBox.information(self, "Info", "Video hasil belum tersedia.")
            return
        dlg = VideoPlayerDialog(self, out_vid, f"Hasil Anotasi - Job #{self.current_job_id}")
        dlg.exec()

    def _get_job_output_video(self, job_id: int) -> Optional[str]:
        jobs = [j for j in self.db.get_all_jobs() if j["id"] == job_id]
        if jobs and jobs[0].get("output_video_path"):
            p = Path(jobs[0]["output_video_path"])
            if p.is_file(): return str(p)
        return None

    def _on_open_output_folder(self) -> None:
        out_dir = self.config.paths.resolved_output_dir()
        try:
            import os
            os.startfile(str(out_dir))
        except Exception as e:
            QMessageBox.warning(self, "Buka Folder", f"Gagal membuka folder:\n{out_dir}\n\nError: {e}")

    def _on_export_csv(self) -> None:
        if not self.current_job_id: return
        containers = self.db.get_full_container_view(self.current_job_id)
        if not containers:
            QMessageBox.information(self, "Ekspor CSV", "Tidak ada data.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan CSV", f"hasil_ocr_job_{self.current_job_id}.csv", "CSV Files (*.csv)")
        if not file_path: return
        import csv
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Region", "Container ID", "Mode / Status", "Nomor Kontainer", "ISO Check", "ISO Code", "Conf", "Manual Review"])
                for c in containers:
                    c_num = c.get("detections", {}).get("container_number", {})
                    iso_d = c.get("detections", {}).get("iso_code", {})
                    num_text = c_num.get("manual_override_text") or c_num.get("ocr_clean_text") or "-"
                    
                    if c_num.get("is_valid_iso"):
                        iso_check = "VALID (Manual)" if c_num.get("is_reviewed") else "VALID (Auto)"
                    else:
                        iso_check = "INVALID"
                        
                    iso_code = iso_d.get("manual_override_text") or iso_d.get("ocr_clean_text") or "-"
                    conf = f"{c_num.get('confidence', 0.0):.2f}"
                    rev = "YES" if c_num.get("is_reviewed") or iso_d.get("is_reviewed") else "NO"
                    writer.writerow([f"R{c['route_order']:03d}", c["container_id"], c.get("container_type", "-"), num_text, iso_check, iso_code, conf, rev])
            QMessageBox.information(self, "Sukses", f"Data diekspor ke:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Gagal", f"Error: {e}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.cmd_queue.put({"action": "UPDATE_CONFIG", "config": self.config.to_dict()})
            self.show_toast(f"Pengaturan disimpan.", "success")

    def _on_logout(self) -> None:
        reply = QMessageBox.question(self, "Logout", "Yakin ingin keluar?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            from src.ui.login_dialog import LoginDialog
            from src.utils.security import store_auth_token
            store_auth_token("")
            self.hide()
            dlg = LoginDialog(self.config, self)
            if dlg.exec() == LoginDialog.DialogCode.Accepted:
                self.show()
                self.show_toast("Berhasil masuk.", "info")
            else:
                self.close()

    def show_toast(self, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        self._toasts.show(message, level, duration_ms)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_toasts", None) is not None:
            self._toasts.reposition()

    def closeEvent(self, event) -> None:
        if self.is_job_running:
            reply = QMessageBox.question(self, "Keluar", "Job berjalan. Batalkan dan keluar?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.orchestrator: self.orchestrator.cancel_current_job()

        if hasattr(self, "event_timer") and self.event_timer.isActive():
            self.event_timer.stop()
        if hasattr(self, "cmd_queue") and self.cmd_queue is not None:
            try: self.cmd_queue.put({"action": "STOP"})
            except Exception: pass
        event.accept()
        app_inst = QApplication.instance()
        if app_inst: app_inst.quit()