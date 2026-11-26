"""
Local ONNX Runtime Inference for Container & Text ROI Detection.
Follows TDD_Container_OCR_Windows_v1.9.md & GEMINI.md:
- Pure CPU execution via onnxruntime.InferenceSession.
- Strict class mapping:
    0: container_left_right
    1: iso_code
    2: container_number
    3: container_rear_front
    4: container_top
- Dynamic spatial resolution support.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.image_io import imread_unicode

CLASS_NAMES = {
    0: "container_left_right",
    1: "iso_code",
    2: "container_number",
    3: "container_rear_front",
    4: "container_top",
}


class DetectionBox(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x": self.x1,
                "y": self.y1,
                "width": self.width,
                "height": self.height,
            },
        }


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Standard CPU Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _cuda_hardware_ready() -> bool:
    """Return True only when an NVIDIA CUDA device is actually usable (not merely compiled into ORT)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_cuda_dlls_loaded() -> None:
    """Register bundled torch/nvidia DLL dirs (frozen .exe) and preload CUDA DLLs before ORT CUDA session."""
    if not _cuda_hardware_ready():
        return
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))
        cand_dirs = [base / "torch" / "lib", base / "onnxruntime" / "capi"]
        nv_root = base / "nvidia"
        if nv_root.is_dir():
            cand_dirs.extend(p / "bin" for p in nv_root.iterdir() if (p / "bin").is_dir())
        for d in cand_dirs:
            if d.is_dir():
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
                ds = str(d)
                if ds not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = ds + os.pathsep + os.environ.get("PATH", "")
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception:
            pass


def resolve_onnx_providers(device: str = "cpu") -> Tuple[List[str], str]:
    """
    Resolve prioritized ONNX Runtime ExecutionProviders from requested device ('cpu', 'auto', 'cuda', 'dml').
    Always falls back cleanly to CPUExecutionProvider if the GPU provider is unavailable.
    """
    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail and not _cuda_hardware_ready():
        avail.discard("CUDAExecutionProvider")
    req = (device or "cpu").strip().lower()
    preferred: List[str] = []
    if req == "cpu":
        preferred = ["CPUExecutionProvider"]
    elif req.startswith("cuda"):
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    elif req == "dml":
        preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:  # "auto" prefers discrete CUDA GPU when available, else fast CPU (avoid slow iGPU DML stalls)
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    providers = [p for p in preferred if p in avail]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers, providers[0]


class OnnxDetector:
    """Wrapper for ONNX Runtime YOLO container & text detection (CPU or GPU CUDA/DirectML)."""

    def __init__(self, model_path: Path | str, device: str = "cpu"):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        self.requested_device = (device or "cpu").strip().lower()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers, _ = resolve_onnx_providers(self.requested_device)
        if "CUDAExecutionProvider" in providers:
            _ensure_cuda_dlls_loaded()
        if "DmlExecutionProvider" in providers:
            opts.enable_mem_pattern = False
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=providers,
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        sess_provs = self.session.get_providers()
        self.active_provider = sess_provs[0] if sess_provs else "CPUExecutionProvider"
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        inp_shape = self.session.get_inputs()[0].shape
        self.model_imgsz: Optional[Tuple[int, int]] = None
        if len(inp_shape) >= 4 and isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int):
            self.model_imgsz = (int(inp_shape[2]), int(inp_shape[3]))

    def detect(
        self,
        image_or_path: np.ndarray | Path | str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.60,
        imgsz: int = 480,
        classes: Optional[List[int]] = None,
    ) -> List[DetectionBox]:
        """Runs detection on an image, returns list of DetectionBox."""
        if isinstance(image_or_path, (str, Path)):
            img = imread_unicode(image_or_path)
            if img is None:
                raise ValueError(f"Failed to read image from {image_or_path}")
        else:
            img = image_or_path

        orig_h, orig_w = img.shape[:2]

        target_shape = self.model_imgsz if self.model_imgsz is not None else (imgsz, imgsz)
        padded_img, ratio, (dw, dh) = letterbox(img, new_shape=target_shape, auto=False)
        blob = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        preds = np.squeeze(outputs, axis=0).T

        boxes = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences >= conf_thresh
        if classes is not None:
            class_mask = np.isin(class_ids, classes)
            mask = mask & class_mask

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(np.round(x1), 0, orig_w).astype(int)
        y1 = np.clip(np.round(y1), 0, orig_h).astype(int)
        x2 = np.clip(np.round(x2), 0, orig_w).astype(int)
        y2 = np.clip(np.round(y2), 0, orig_h).astype(int)

        scaled_boxes = np.stack([x1, y1, x2, y2], axis=1)

        final_boxes: List[DetectionBox] = []
        unique_classes = np.unique(class_ids)

        for cid in unique_classes:
            c_mask = class_ids == cid
            c_boxes = scaled_boxes[c_mask]
            c_scores = confidences[c_mask]

            keep_indices = nms_cpu(c_boxes, c_scores, iou_threshold=iou_thresh)
            for idx in keep_indices:
                box = c_boxes[idx]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                final_boxes.append(
                    DetectionBox(
                        class_id=int(cid),
                        class_name=CLASS_NAMES.get(int(cid), f"class_{cid}"),
                        confidence=float(c_scores[idx]),
                        x1=int(box[0]),
                        y1=int(box[1]),
                        x2=int(box[2]),
                        y2=int(box[3]),
                    )
                )

        return final_boxes

    @staticmethod
    def crop_box(image: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
        """Crops a region given {'x', 'y', 'width', 'height'}."""
        x = max(0, bbox["x"])
        y = max(0, bbox["y"])
        w = max(1, bbox["width"])
        h = max(1, bbox["height"])
        img_h, img_w = image.shape[:2]
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        return image[y:y2, x:x2]


# ---------------------------------------------------------------------------
# Compute plan: cached, since torch/CUDA/ONNX provider probing is expensive
# (cold start 1–3 s). Hardware doesn't change during an app run, so caching
# per requested device is safe. Callers get a fresh dict copy so mutation
# (e.g. run_gpu_preflight_probe) cannot corrupt the cache.
# ---------------------------------------------------------------------------

def describe_compute_plan_impl(device: str = "auto") -> Dict[str, Any]:
    """
    Summarize exact runtime compute device per pipeline stage without loading models.
    Reused by SettingsDialog preview and pre-run MainWindow confirmation.
    """
    req = (device or "cpu").strip().lower()
    avail = set(ort.get_available_providers())
    _, onnx_prov = resolve_onnx_providers(req)

    cuda_avail = False
    cuda_name = ""
    torch_ver = "unknown"
    try:
        import torch
        torch_ver = str(torch.__version__)
        cuda_avail = bool(torch.cuda.is_available())
        if cuda_avail:
            cuda_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    has_dml = "DmlExecutionProvider" in avail
    has_cuda_ort = ("CUDAExecutionProvider" in avail) and cuda_avail

    if req in ("auto", "cuda") and cuda_avail:
        stitch_dev = f"GPU NVIDIA CUDA ({cuda_name or 'cuda:0'})"
        track_prov = "CUDAExecutionProvider" if has_cuda_ort else ("DmlExecutionProvider" if has_dml else "CPUExecutionProvider")
    elif req in ("dml", "cuda") and has_dml:
        stitch_dev = "CPU (PyTorch — DirectML hanya untuk ONNX)"
        track_prov = "DmlExecutionProvider"
    else:
        stitch_dev = "CPU (PyTorch)"
        track_prov = onnx_prov

    if req == "auto" and not cuda_avail and has_dml:
        hint = (
            "Mode 'Auto' memakai CPU karena GPU NVIDIA CUDA tidak tersedia "
            "(pada iGPU laptop, CPU sering lebih cepat daripada DirectML). "
            "Pilih 'GPU DirectML ONNX' di Pengaturan jika ingin memaksa GPU pada Track Frame & Deteksi."
        )
    elif req == "auto" and not cuda_avail:
        hint = "Mode 'Auto' memakai CPU karena GPU NVIDIA CUDA tidak tersedia pada PC ini."
    elif req.startswith("cuda") and not cuda_avail and has_dml:
        hint = "GPU NVIDIA CUDA tidak tersedia; Track Frame & Deteksi dialihkan ke GPU DirectML (DmlExecutionProvider)."
    elif req.startswith("cuda") and not cuda_avail:
        hint = "GPU NVIDIA CUDA tidak terdeteksi pada PC ini; semua tahap otomatis memakai CPU."
    elif req == "dml" and not has_dml:
        hint = "DmlExecutionProvider tidak tersedia pada runtime ini; fallback aman ke CPUExecutionProvider."
    elif req == "cpu":
        hint = "Semua tahap dikunci ke CPU sesuai pilihan pengguna."
    else:
        hint = ""

    return {
        "requested": req,
        "cuda_available": cuda_avail,
        "cuda_name": cuda_name,
        "torch_version": torch_ver,
        "has_dml": has_dml,
        "available_providers": sorted(avail),
        "track_provider": track_prov,
        "stitch_device": stitch_dev,
        "onnx_provider": onnx_prov,
        "uses_gpu": (track_prov != "CPUExecutionProvider") or stitch_dev.startswith("GPU"),
        "hint": hint,
    }


@lru_cache(maxsize=8)
def _describe_compute_plan_cached(device_key: str) -> Dict[str, Any]:
    return describe_compute_plan_impl(device_key)


def describe_compute_plan(device: str = "auto") -> Dict[str, Any]:
    """
    Cached wrapper around describe_compute_plan_impl.

    Returns a shallow copy so callers can freely mutate the result
    (e.g. run_gpu_preflight_probe adds 'probe_ok', 'active_provider', ...)
    without polluting the cache. Values inside are primitives or lists that
    callers do not modify.
    """
    key = str(device or "cpu").strip().lower()
    return dict(_describe_compute_plan_cached(key))


def clear_compute_plan_cache() -> None:
    """Clear cache — useful if hardware state changes (rare) or in unit tests."""
    _describe_compute_plan_cached.cache_clear()


def run_gpu_preflight_probe(model_path: Path | str, device: str = "auto") -> Dict[str, Any]:
    """Run a quick live 480x480 warmup + timed inference check for the requested compute device."""
    import time

    # Copy — do not mutate the cached plan dict.
    plan = describe_compute_plan(device)
    m_path = Path(model_path)
    if not m_path.is_file():
        plan["probe_ok"] = False
        plan["probe_error"] = f"File model ONNX tidak ditemukan: {m_path}"
        return plan

    dummy = np.zeros((480, 480, 3), dtype=np.uint8)
    try:
        t0 = time.perf_counter()
        det = OnnxDetector(m_path, device=plan["requested"])
        _ = det.detect(dummy, conf_thresh=0.5, imgsz=480, classes=[0])
        t1 = time.perf_counter()
        _ = det.detect(dummy, conf_thresh=0.5, imgsz=480, classes=[0])
        infer_ms = round((time.perf_counter() - t1) * 1000.0, 1)
        warmup_ms = round((t1 - t0) * 1000.0, 1)

        cpu_ms: Optional[float] = None
        if det.active_provider != "CPUExecutionProvider":
            det_cpu = OnnxDetector(m_path, device="cpu")
            _ = det_cpu.detect(dummy, conf_thresh=0.5, imgsz=480, classes=[0])
            tc = time.perf_counter()
            _ = det_cpu.detect(dummy, conf_thresh=0.5, imgsz=480, classes=[0])
            cpu_ms = round((time.perf_counter() - tc) * 1000.0, 1)

        plan.update({
            "probe_ok": True,
            "active_provider": det.active_provider,
            "warmup_ms": warmup_ms,
            "infer_ms": infer_ms,
            "cpu_infer_ms": cpu_ms,
        })
    except Exception as exc:
        plan["probe_ok"] = False
        plan["probe_error"] = str(exc)
    return plan