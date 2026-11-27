"""
Settings Dialog for Container OCR Desktop.
Configures:
- Halotec OCR API: URL, Gate ("1"), Bearer Token (stored via Windows DPAPI).
- Detection & Stitching Thresholds.
- Test connection utility.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.ocr_client.client import HalotecOcrClient
from src.recognition.detector import (
    describe_compute_plan,
    get_hardware_summary,
    run_gpu_preflight_probe,
)
from src.utils.security import retrieve_auth_token, store_auth_token


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _ProbeSignals(QObject):
    ready = Signal(str, str)     # (hw_text, compute_preview_html)
    progress = Signal(str)        # status text — Solusi 4


class _ProbeWorker(QRunnable):
    """Runs torch/CUDA/ONNX hardware probing off the UI thread."""

    def __init__(self, requested_device: str, signals: _ProbeSignals):
        super().__init__()
        self.requested_device = requested_device
        self.signals = signals

    def run(self) -> None:
        try:
            self.signals.progress.emit("Memuat PyTorch & cek CUDA…")
            hw_text = SettingsDialog._detect_hardware_summary()
            self.signals.progress.emit("Menyusun compute plan…")
            plan = describe_compute_plan(self.requested_device)
            preview_html = SettingsDialog._format_compute_preview(plan)
        except Exception as e:
            hw_text = f"Status Hardware: gagal dideteksi ({e})"
            preview_html = f"<i>Gagal menghitung compute plan: {e}</i>"
        self.signals.ready.emit(hw_text, preview_html)


class _PreflightSignals(QObject):           # ← typo "Prefilght" diperbaiki
    ready = Signal(dict)
    error = Signal(str)


class _PreflightWorker(QRunnable):
    """Menjalankan run_gpu_preflight_probe() di background agar UI tidak freeze."""

    def __init__(self, model_path, device: str, signals: _PreflightSignals):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.signals = signals

    def run(self) -> None:
        try:
            res = run_gpu_preflight_probe(self.model_path, self.device)
            self.signals.ready.emit(res)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config

        # --- Probe hardware (Solusi 2 + 4) ---
        self._probe_ready = False
        self._probe_started = False
        self._probe_signals = _ProbeSignals(self)
        self._probe_signals.ready.connect(self._on_probe_ready)
        self._probe_signals.progress.connect(self._on_probe_progress)

        # --- Preflight GPU test (Solusi 1) — INI YANG HILANG DI KODE ANDA ---
        self._preflight_running = False
        self._preflight_signals = _PreflightSignals(self)
        self._preflight_signals.ready.connect(self._on_preflight_ready)
        self._preflight_signals.error.connect(self._on_preflight_error)

        self.setWindowTitle("Pengaturan Aplikasi & OCR")
        self.resize(550, 520)
        self._init_ui()

        # Start the probe AFTER the dialog paints — UI feels instant.
        QTimer.singleShot(0, self._start_hw_probe)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._probe_started:
            self._start_hw_probe()

    def _start_hw_probe(self) -> None:
        # FIX BUG #4: guard hanya pakai _probe_started, bukan AND _probe_ready.
        if self._probe_started:
            return
        self._probe_started = True
        dev = str(self.device_combo.currentData() or "auto")
        QThreadPool.globalInstance().start(_ProbeWorker(dev, self._probe_signals))

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. OCR API Settings
        api_group = QGroupBox("Halotec OCR Service API")
        api_form = QFormLayout(api_group)

        self.api_url_edit = QLineEdit(self.config.api.url)
        api_form.addRow("API Base URL:", self.api_url_edit)

        self.gate_edit = QLineEdit(self.config.api.gate)
        self.gate_edit.setToolTip("Wajib string angka, misalnya '1' (non-angka memicu error backend 500)")
        api_form.addRow("Gate ID (Angka):", self.gate_edit)

        current_token = retrieve_auth_token() or ""
        self.token_edit = QLineEdit(current_token)
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("JWT Bearer Token (disimpan aman di Windows DPAPI)")

        token_row = QHBoxLayout()
        token_row.addWidget(self.token_edit)

        self.btn_show_token = QPushButton("Lihat")
        self.btn_show_token.setCheckable(True)
        self.btn_show_token.toggled.connect(
            lambda checked: self.token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        token_row.addWidget(self.btn_show_token)
        api_form.addRow("Auth Token:", token_row)

        test_row = QHBoxLayout()
        self.btn_test_conn = QPushButton("Uji Token Aktif")
        self.btn_test_conn.clicked.connect(self._test_connection)
        test_row.addWidget(self.btn_test_conn)
        api_form.addRow("", test_row)

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Username / Email akun OCR")
        api_form.addRow("Login Akun:", self.user_edit)

        pwd_row = QHBoxLayout()
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pwd_edit.setPlaceholderText("Password akun OCR")
        pwd_row.addWidget(self.pwd_edit)

        self.btn_login = QPushButton("Login & Dapatkan Token")
        self.btn_login.setStyleSheet("background-color: #0078D4; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_login.clicked.connect(self._do_login)
        pwd_row.addWidget(self.btn_login)
        api_form.addRow("Password:", pwd_row)

        layout.addWidget(api_group)

        # 2. Detection & Compute Device Settings
        det_group = QGroupBox("Model, Perangkat Komputasi && Deteksi AI")
        det_form = QFormLayout(det_group)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU NVIDIA CUDA jika ada, else CPU Optimal)", "auto")
        self.device_combo.addItem("CPU (Stabil & Kompatibel Semua PC)", "cpu")
        avail_ort: set[str] = set()
        try:
            import onnxruntime as _ort_ui
            avail_ort = set(_ort_ui.get_available_providers())
        except Exception:
            pass
        if "DmlExecutionProvider" in avail_ort:
            self.device_combo.addItem("GPU NVIDIA CUDA (fallback DirectML / CPU)", "cuda")
            self.device_combo.addItem("GPU DirectML ONNX (AMD / Intel / NVIDIA)", "dml")
        else:
            self.device_combo.addItem("GPU NVIDIA CUDA (fallback CPU)", "cuda")
        curr_dev = str(getattr(self.config.stitching, "device", "auto") or "auto").lower()
        dev_idx = self.device_combo.findData(curr_dev)
        self.device_combo.setCurrentIndex(dev_idx if dev_idx >= 0 else 0)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        det_form.addRow("Perangkat Komputasi:", self.device_combo)

        hw_row = QHBoxLayout()
        self.lbl_hw = QLabel("Mendeteksi hardware…")
        self.lbl_hw.setStyleSheet("color: #aaa; font-size: 11px;")
        self.lbl_hw.setWordWrap(True)
        hw_row.addWidget(self.lbl_hw, 1)

        self.btn_test_gpu = QPushButton("Cek && Uji GPU")
        self.btn_test_gpu.setToolTip("Jalankan uji inferensi ONNX & cek CUDA/DirectML sebelum memproses video")
        self.btn_test_gpu.clicked.connect(self._on_test_gpu_clicked)
        hw_row.addWidget(self.btn_test_gpu)
        det_form.addRow("Status Hardware:", hw_row)

        self.lbl_compute_preview = QLabel("Menghitung compute plan…")
        self.lbl_compute_preview.setWordWrap(True)
        self.lbl_compute_preview.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #1e1e1e; border-radius: 4px;"
        )
        det_form.addRow("Aktif Sebelum Run:", self.lbl_compute_preview)

        self.thresh_container = QDoubleSpinBox()
        self.thresh_container.setRange(0.05, 0.99)
        self.thresh_container.setSingleStep(0.05)
        self.thresh_container.setValue(self.config.detection.threshold_container)
        det_form.addRow("Threshold Kontainer:", self.thresh_container)

        self.thresh_text = QDoubleSpinBox()
        self.thresh_text.setRange(0.05, 0.99)
        self.thresh_text.setSingleStep(0.05)
        self.thresh_text.setValue(self.config.detection.threshold_text)
        det_form.addRow("Threshold Nomor & ISO:", self.thresh_text)

        self.sample_interval_spin = QDoubleSpinBox()
        self.sample_interval_spin.setRange(0.1, 10.0)
        self.sample_interval_spin.setSingleStep(0.5)
        self.sample_interval_spin.setDecimals(1)
        self.sample_interval_spin.setValue(getattr(self.config.detection, "frame_sample_interval_seconds", 2.0))
        self.sample_interval_spin.setSuffix(" detik")
        det_form.addRow("Interval Sampling Frame (Detik):", self.sample_interval_spin)

        self.min_crops = QSpinBox()
        self.min_crops.setRange(5, 200)
        self.min_crops.setValue(self.config.stitching.min_crops_per_track)
        det_form.addRow("Min Crops per Track (Stitching):", self.min_crops)

        layout.addWidget(det_group)

        # 3. Video Output Settings
        video_group = QGroupBox("Pengaturan Output Video Hasil Anotasi")
        video_form = QFormLayout(video_group)

        self.video_res_combo = QComboBox()
        self.video_res_combo.addItem("720p (1280x720) - Ringan & Cepat (Direkomendasikan)", "720P")
        self.video_res_combo.addItem("1080p (1920x1080) - Full HD Tajam", "1080P")
        self.video_res_combo.addItem("Asli / Original (Resolusi Penuh Drone / 4K)", "ORIGINAL")

        curr_res = getattr(self.config, "video_resolution", "720P").upper()
        idx = self.video_res_combo.findData(curr_res)
        self.video_res_combo.setCurrentIndex(idx if idx >= 0 else 0)

        video_form.addRow("Resolusi Video:", self.video_res_combo)
        layout.addWidget(video_group)

        # 4. Path Settings
        path_group = QGroupBox("Lokasi Output & Data (Option 2 Portable)")
        path_form = QFormLayout(path_group)

        self.out_dir_edit = QLineEdit(self.config.paths.output_dir)
        path_form.addRow("Folder Output:", self.out_dir_edit)

        layout.addWidget(path_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ----------------------------------------------------- Hardware probe

    @staticmethod
    def _detect_hardware_summary() -> str:
        hw = get_hardware_summary()   # cache hit setelah probe pertama
        if hw["cuda_available"]:
            cuda_info = f"CUDA: Tersedia ({hw['cuda_name']})"
        else:
            cuda_info = f"CUDA: Tidak tersedia (torch {hw['torch_version']})"
        dml_note = " | DirectML GPU: Aktif" if hw["has_dml"] else ""
        ort_info = ", ".join(hw["available_providers"])
        return f"{cuda_info}{dml_note} | ONNX: [{ort_info}]"

    @staticmethod
    def _format_compute_preview(plan: dict) -> str:
        track_is_gpu = plan["track_provider"] != "CPUExecutionProvider"
        color = "#4ec9b0" if track_is_gpu else "#dcdcaa"
        lines = [
            f"• <b>Tracking Frame (Tahap 1) &amp; Deteksi 4K/FRONT_REAR:</b> {plan['track_provider']}",
            f"• <b>Stitching (ALIKED/LightGlue):</b> {plan['stitch_device']}",
        ]
        if plan.get("hint"):
            lines.append(f"• <i>{plan['hint']}</i>")
        return (
            f'<span style="color: {color}; font-size: 11px;">'
            + "<br>".join(lines)
            + "</span>"
        )

    def _on_probe_progress(self, text: str) -> None:
        if not self._probe_ready:
            self.lbl_hw.setText(f"⏳ {text}")

    def _on_probe_ready(self, hw_text: str, preview_html: str) -> None:
        self._probe_ready = True
        self.lbl_hw.setText(hw_text)
        try:
            sel_dev = str(self.device_combo.currentData() or "auto")
            plan = describe_compute_plan(sel_dev)
            self.lbl_compute_preview.setText(self._format_compute_preview(plan))
        except Exception:
            self.lbl_compute_preview.setText(preview_html)

    def _on_device_changed(self, *_args) -> None:
        if not self._probe_ready:
            return  # tunggu probe selesai agar tidak menyentuh torch di UI thread
        try:
            sel_dev = str(self.device_combo.currentData() or "auto")
            plan = describe_compute_plan(sel_dev)
            self.lbl_compute_preview.setText(self._format_compute_preview(plan))
        except Exception as e:
            self.lbl_compute_preview.setText(f"<i>Gagal menghitung compute plan: {e}</i>")

    def _update_compute_preview(self) -> None:
        self._on_device_changed()

    # --------------------------------------------------------- GPU probe

    def _on_test_gpu_clicked(self) -> None:
        if self._preflight_running:
            return  # cegah double-klik

        sel_dev = str(self.device_combo.currentData() or "auto")
        model_p = self.config.detection.resolved_model_path()

        self._preflight_running = True
        self.btn_test_gpu.setEnabled(False)
        self.btn_test_gpu.setText("⏳ Menguji…")
        self.lbl_hw.setText(f"Menguji {sel_dev.upper()} — memuat model & warmup inference…")

        QThreadPool.globalInstance().start(
            _PreflightWorker(model_p, sel_dev, self._preflight_signals)
        )

    def _on_preflight_ready(self, res: dict) -> None:
        self._preflight_running = False
        self.btn_test_gpu.setEnabled(True)
        self.btn_test_gpu.setText("Cek && Uji GPU")

        # Refresh label hardware (cache hit — get_hardware_summary sudah dipanggil worker)
        try:
            self._probe_ready = True
            self.lbl_hw.setText(self._detect_hardware_summary())
            sel_dev = str(self.device_combo.currentData() or "auto")
            plan = describe_compute_plan(sel_dev)
            self.lbl_compute_preview.setText(self._format_compute_preview(plan))
        except Exception:
            pass

        if not res.get("probe_ok"):
            QMessageBox.warning(
                self, "Cek GPU Gagal",
                f"Gagal menguji model ONNX:\n{res.get('probe_error', 'Error tidak diketahui')}",
            )
            return

        speed_line = (
            f"• Latensi ONNX ({res['active_provider']}): "
            f"{res['infer_ms']} ms/frame (warmup {res['warmup_ms']} ms)"
        )
        if res.get("cpu_infer_ms") is not None:
            speed_line += f"\n• Pembanding CPU (CPUExecutionProvider): {res['cpu_infer_ms']} ms/frame"

        msg = (
            f"Mode Dipilih: {str(self.device_combo.currentData() or 'auto').upper()}\n\n"
            f"• Tracking Frame & Deteksi ONNX: {res['active_provider']}\n"
            f"• Stitching ALIKED/LightGlue: {res['stitch_device']}\n"
            f"{speed_line}\n"
        )
        if res.get("hint"):
            msg += f"\nCatatan:\n{res['hint']}"
        QMessageBox.information(self, "Hasil Cek & Uji Perangkat GPU", msg)

    def _on_preflight_error(self, err: str) -> None:
        self._preflight_running = False
        self.btn_test_gpu.setEnabled(True)
        self.btn_test_gpu.setText("Cek && Uji GPU")
        QMessageBox.critical(self, "Cek GPU Gagal", f"Terjadi error tak terduga:\n{err}")

    # ------------------------------------------------------------ Login

    def _do_login(self) -> None:
        url = self.api_url_edit.text().strip()
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Peringatan", "Silakan masukkan username dan password akun OCR!")
            return

        client = HalotecOcrClient(base_url=url)
        try:
            token = client.login(username=username, password=password)
            self.token_edit.setText(token)
            QMessageBox.information(
                self, "Login Berhasil",
                "Berhasil login ke Halotec OCR Service!\nToken baru telah tersimpan dan siap digunakan.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Login Gagal", f"Gagal login ke server OCR:\n{e}")

    def _test_connection(self) -> None:
        url = self.api_url_edit.text().strip()
        gate = self.gate_edit.text().strip()
        token = self.token_edit.text().strip()

        if not gate.isdigit():
            QMessageBox.warning(self, "Peringatan", "Gate ID harus berupa angka (misal '1')!")
            return

        if not token:
            QMessageBox.warning(
                self, "Token Kosong",
                "Kolom Auth Token masih kosong.\nSilakan paste token yang valid atau login dengan username & password di bawah.",
            )
            return

        client = HalotecOcrClient(base_url=url, gate=gate)
        client.set_token(token)

        try:
            new_token = client.refresh_token()
            if new_token:
                self.token_edit.setText(new_token)
                QMessageBox.information(
                    self, "Sukses",
                    "Koneksi ke Halotec OCR Service berhasil!\nToken aktif dan berhasil di-refresh.",
                )
            else:
                QMessageBox.warning(
                    self, "Token Tidak Valid / Kedaluwarsa",
                    "Server OCR dapat dihubungi, namun token tidak valid atau sudah kedaluwarsa.\nSilakan masukkan token baru atau login ulang.",
                )
        except Exception as e:
            QMessageBox.critical(self, "Koneksi Gagal", f"Gagal menghubungi server OCR:\n{e}")

    def _save_and_accept(self) -> None:
        url = self.api_url_edit.text().strip()
        gate = self.gate_edit.text().strip()
        token = self.token_edit.text().strip()

        if not gate.isdigit():
            QMessageBox.warning(self, "Validasi Gagal", "Gate ID harus berupa angka murni (contoh: '1')!")
            return

        self.config.api.url = url
        self.config.api.gate = gate
        self.config.detection.threshold_container = float(self.thresh_container.value())
        self.config.detection.threshold_text = float(self.thresh_text.value())
        self.config.detection.frame_sample_interval_seconds = float(self.sample_interval_spin.value())
        self.config.stitching.device = str(self.device_combo.currentData() or "cpu")
        self.config.stitching.min_crops_per_track = int(self.min_crops.value())
        self.config.video_resolution = self.video_res_combo.currentData() or "720P"
        self.config.paths.output_dir = self.out_dir_edit.text().strip()

        curr_saved_token = retrieve_auth_token() or ""
        clean_token = token.strip()
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()
        clean_token = clean_token.strip("'\"")

        if clean_token and clean_token != curr_saved_token:
            client = HalotecOcrClient(base_url=url, gate=gate)
            is_valid, msg, validated_token = client.validate_token(clean_token)
            if not is_valid:
                QMessageBox.warning(
                    self, "Token Tidak Valid",
                    f"Pengaturan tidak dapat disimpan karena token ditolak oleh server:\n\n{msg}\n\nSilakan periksa kembali atau login dengan akun Anda.",
                )
                return
            store_auth_token(validated_token or clean_token)
        elif not clean_token:
            store_auth_token("")

        self.config.save()
        self.accept()