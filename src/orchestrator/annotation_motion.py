from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


from src.video.ffmpeg_writer import FFmpegWriter


Box = Tuple[float, float, float, float]


def box_iou(boxA: Box, boxB: Box) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea <= 0.0:
        return 0.0
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(areaA + areaB - interArea)

def _build_banner_cache(overlays:list,sf:float,fscale:float,fthick:int,color_container):
    """Pre-render semua banner label sekali per render call."""
    cache: dict[str, "np.ndarray"] = {}
    tag_h = max(1, int(18 * sf))
    for ov in overlays:
        tag_w = max(1, int((len(ov.label) * 8.5 + 10) * sf))
        sprite = np.zeros((tag_h, tag_w, 3), dtype=np.uint8)
        sprite[:] = color_container
        cv2.putText(
            sprite, ov.label,
            (int(3 * sf), max(int(13 * sf), tag_h - int(4 * sf))),
            cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), fthick, cv2.LINE_AA,
        )
        cache[ov.label] = sprite
    return cache


# --------------------------------------------------------------------------
# 1. Global camera-motion table (pure translation, scale-drift free)
# --------------------------------------------------------------------------
def _lk_translation(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_corners: int = 250,
    win_size: int = 21,
    quality_level: float = 0.01,
    min_distance: int = 6,
) -> Optional[Tuple[float, float]]:
    """Estimate sub-pixel translation (dx, dy) in prev_gray pixels via Forward-Backward LK."""
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, mask=mask
    )
    if pts is None or len(pts) < 6:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts, None, winSize=(win_size, win_size), maxLevel=4
    )
    if nxt is None or st is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray, prev_gray, nxt, None, winSize=(win_size, win_size), maxLevel=4
    )
    if back is None or st2 is None:
        return None
    fb_err = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)
    if int(good.sum()) < 4:
        return None
    diff = (nxt[good] - pts[good]).reshape(-1, 2)
    med = np.median(diff, axis=0)
    inl = diff[np.linalg.norm(diff - med, axis=1) <= 1.2]
    d = np.mean(inl, axis=0) if len(inl) >= 3 else med
    dx, dy = float(d[0]), float(d[1])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    return dx, dy


def estimate_cumulative_motion(
    video_path: str,
    work_width: int = 640,
    max_corners: int = 250,
    cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """
    Hitung tabel transformasi kamera kumulatif C[f] (translasi murni 3x3 tanpa scale shrink).
    C[f] @ pt_frame0 = pt_frame_f (koordinat piksel asli).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((1, 3, 3), dtype=np.float64)

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    hs = max(2, int(round(work_width * h0 / float(max(1, w0)))))
    sx = work_width / float(max(1, w0))
    sy = hs / float(max(1, h0))

    def small_gray(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if g.shape[1] != work_width:
            g = cv2.resize(g, (work_width, hs), interpolation=cv2.INTER_AREA)
        return g

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return np.array([np.eye(3, dtype=np.float64)])

    prev = small_gray(first)
    cum: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    cum_dx, cum_dy = 0.0, 0.0
    last_step_x, last_step_y = 0.0, 0.0
    idx = 1

    while True:
        if cancel is not None and cancel():
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cur = small_gray(frame)
        delta = _lk_translation(prev, cur, max_corners=max_corners)
        if delta is not None:
            step_x = delta[0] / sx
            step_y = delta[1] / sy
            last_step_x, last_step_y = step_x, step_y
        else:
            step_x, step_y = last_step_x * 0.5, last_step_y * 0.5

        cum_dx += step_x
        cum_dy += step_y
        M = np.eye(3, dtype=np.float64)
        M[0, 2] = cum_dx
        M[1, 2] = cum_dy
        cum.append(M)
        prev = cur
        idx += 1
        if progress_callback and total > 0 and idx % 30 == 0:
            progress_callback(min(1.0, idx / float(total)), f"Estimasi gerakan kamera ({idx}/{total})...")

    cap.release()
    return np.asarray(cum, dtype=np.float64)


def map_box(box: Box, C: np.ndarray, a: int, b: int) -> Box:
    """Proyeksikan bounding box dari frame a ke frame b menggunakan tabel C."""
    if a == b:
        return box
    a = int(np.clip(a, 0, len(C) - 1))
    b = int(np.clip(b, 0, len(C) - 1))
    M = C[b] @ np.linalg.inv(C[a])
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=float).T
    q = M @ pts
    return float(q[0].min()), float(q[1].min()), float(q[0].max()), float(q[1].max())


# --------------------------------------------------------------------------
# 2. Overlay per kontainer
# --------------------------------------------------------------------------
@dataclass
class Overlay:
    label: str
    is_valid: bool
    anchors: Dict[int, Box]                             # frame 0-based -> box nomor terverifikasi
    first_frame: int                                    # batas awal tampil
    last_frame: int                                     # batas akhir tampil
    golden_frame: Optional[int] = None                  # indeks frame golden (0-based)
    golden_container_box: Optional[Box] = None          # bounding box badan kontainer di golden frame
    container_history: Dict[int, Box] = field(default_factory=dict)  # frame 0-based -> box kontainer
    container_anchors: Dict[int, Box] = field(default_factory=dict)  # frame 0-based -> box kontainer unclipped
    rel_badge: Optional[Tuple[float, float, float, float]] = None
    max_extrapolation: int = 120
    min_number_visible: float = 0.50
    min_container_visible: float = 0.15
    is_stitching: bool = False
    _keys: List[int] = field(default_factory=list, repr=False)
    _cont_keys: List[int] = field(default_factory=list, repr=False)
    _cont_anchor_keys: List[int] = field(default_factory=list, repr=False)
    _resolved_cont: Dict[int, Box] = field(default_factory=dict, repr=False)
    _resolved_num: Dict[int, Box] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._keys = sorted(self.anchors)
        self._cont_keys = sorted(self.container_history)
        self._cont_anchor_keys = sorted(self.container_anchors)

    @staticmethod
    def _box_center_in_view(b: Box, w0: int, margin_frac: float = 0.02) -> bool:
        if b[2] <= 0 or b[0] >= w0:
            return False
        cx = (b[0] + b[2]) / 2.0
        return (margin_frac * w0) <= cx <= ((1.0 - margin_frac) * w0)

    @staticmethod
    def _box_has_screen_presence(b: Box, w0: int, h0: int, min_px: float = 80.0) -> bool:
        ix = min(b[2], float(w0)) - max(b[0], 0.0)
        iy = min(b[3], float(h0)) - max(b[1], 0.0)
        if ix <= 0.0 or iy <= 0.0:
            return False
        bw = max(1.0, b[2] - b[0])
        bh = max(1.0, b[3] - b[1])
        return ix >= min(min_px, 0.25 * bw) and iy >= min(min_px, 0.25 * bh)

    def _interp_dict_with_C(
        self, table: Dict[int, Box], keys: List[int], f: int, C: Optional[np.ndarray]
    ) -> Optional[Box]:
        if not keys:
            return None
        i = bisect_left(keys, f)
        if i == 0:
            a = keys[0]
            return map_box(table[a], C, a, f) if C is not None and len(C) > 0 else table[a]
        if i == len(keys):
            a = keys[-1]
            return map_box(table[a], C, a, f) if C is not None and len(C) > 0 else table[a]
        a0, a1 = keys[i - 1], keys[i]
        if a1 == a0:
            return table[a0]
        b0 = map_box(table[a0], C, a0, f) if C is not None and len(C) > 0 else table[a0]
        b1 = map_box(table[a1], C, a1, f) if C is not None and len(C) > 0 else table[a1]
        w = (f - a0) / float(a1 - a0)
        return (
            (1.0 - w) * b0[0] + w * b1[0],
            (1.0 - w) * b0[1] + w * b1[1],
            (1.0 - w) * b0[2] + w * b1[2],
            (1.0 - w) * b0[3] + w * b1[3],
        )

    def container_box_at(self, f: int, C: Optional[np.ndarray], w0: int = 1920, h0: int = 1080) -> Optional[Box]:
        """Mengembalikan bounding box badan kontainer pada frame f."""
        if self._resolved_cont:
            cb = self._resolved_cont.get(f)
            if cb is None:
                return None
            if not self._box_has_screen_presence(cb, w0, h0, 90.0):
                return None
            return cb

        if self._cont_keys:
            c_min, c_max = self._cont_keys[0], self._cont_keys[-1]
            if f < c_min - self.max_extrapolation or f > c_max + self.max_extrapolation:
                return None
            i = bisect_left(self._cont_keys, f)
            if i == 0:
                if C is not None and len(C) > 0:
                    cb = map_box(self.container_history[self._cont_keys[0]], C, self._cont_keys[0], f)
                elif len(self._cont_keys) > 1:
                    f0, f1 = self._cont_keys[0], self._cont_keys[1]
                    b0, b1 = self.container_history[f0], self.container_history[f1]
                    rate = (f - f0) / float(max(1, f1 - f0))
                    cb = tuple(p0 + rate * (p1 - p0) for p0, p1 in zip(b0, b1))  # type: ignore[assignment]
                else:
                    cb = self.container_history[self._cont_keys[0]]
            elif i == len(self._cont_keys):
                if C is not None and len(C) > 0:
                    cb = map_box(self.container_history[self._cont_keys[-1]], C, self._cont_keys[-1], f)
                elif len(self._cont_keys) > 1:
                    f0, f1 = self._cont_keys[-2], self._cont_keys[-1]
                    b0, b1 = self.container_history[f0], self.container_history[f1]
                    rate = (f - f1) / float(max(1, f1 - f0))
                    cb = tuple(p1 + rate * (p1 - p0) for p0, p1 in zip(b0, b1))  # type: ignore[assignment]
                else:
                    cb = self.container_history[self._cont_keys[-1]]
            else:
                f0, f1 = self._cont_keys[i - 1], self._cont_keys[i]
                if f1 == f0:
                    cb = self.container_history[f0]
                else:
                    b0, b1 = self.container_history[f0], self.container_history[f1]
                    w = (f - f0) / float(f1 - f0)
                    cb = tuple((1.0 - w) * p0 + w * p1 for p0, p1 in zip(b0, b1))  # type: ignore[assignment]
            if not self._box_has_screen_presence(cb, w0, h0, 90.0):
                return None
            return (float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))

        if self._cont_anchor_keys:
            min_a, max_a = self._cont_anchor_keys[0], self._cont_anchor_keys[-1]
            if f < min_a - self.max_extrapolation or f > max_a + self.max_extrapolation:
                return None
            cb = self._interp_dict_with_C(self.container_anchors, self._cont_anchor_keys, f, C)
            if cb is None or not self._box_has_screen_presence(cb, w0, h0, 90.0):
                return None
            return cb

        if self.golden_container_box is not None and self.golden_frame is not None and C is not None and len(C) > 0:
            min_a = self._keys[0] if self._keys else (self.golden_frame - 30)
            max_a = self._keys[-1] if self._keys else (self.golden_frame + 30)
            if f < min_a - self.max_extrapolation or f > max_a + self.max_extrapolation:
                return None
            cont_box = map_box(self.golden_container_box, C, self.golden_frame, f)
            if not self._box_has_screen_presence(cont_box, w0, h0, 90.0):
                return None
            return cont_box

        return None

    def box_at(self, f: int, C: Optional[np.ndarray], w0: int = 1920, h0: int = 1080) -> Optional[Box]:
        # Path 0: ROI-local LK resolved coordinates (computed during render_overlay_video)
        if self._resolved_num:
            b = self._resolved_num.get(f)
            if b is None:
                return None
            if _visible_fraction(b, w0, h0) < self.min_number_visible:
                return None
            if not self._box_center_in_view(b, w0, margin_frac=0.002):
                return None
            return b

        # Path A: Container-Centric Pinned Tracking when anchors are absent (unit tests / rigid fallback)
        if self._cont_keys and self.rel_badge is not None and not self._keys:
            cb = self.container_box_at(f, C, w0, h0)
            if cb is None:
                return None
            cw = max(1.0, float(cb[2] - cb[0]))
            ch = max(1.0, float(cb[3] - cb[1]))
            rx1, ry1, rx2, ry2 = self.rel_badge
            b = (
                float(cb[0] + rx1 * cw),
                float(cb[1] + ry1 * ch),
                float(cb[0] + rx2 * cw),
                float(cb[1] + ry2 * ch),
            )
            if not self._box_center_in_view(b, w0):
                return None
            return b

        # Path B: Multi-anchor interpolation + C-warping
        if not self._keys:
            return None

        min_a, max_a = self._keys[0], self._keys[-1]
        if f < min_a - self.max_extrapolation or f > max_a + self.max_extrapolation:
            return None

        cb = self.container_box_at(f, C, w0, h0)
        if (self.golden_container_box is not None or self._cont_anchor_keys) and cb is None:
            return None

        if C is not None and len(C) > 0:
            b = self._interp_dict_with_C(self.anchors, self._keys, f, C)
        else:
            b = self._interp_dict_with_C(self.anchors, self._keys, f, None)

        if b is None or _visible_fraction(b, w0, h0) < self.min_number_visible:
            return None
        if not self._box_center_in_view(b, w0):
            return None
        return b


def _visible_fraction(box: Box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    ix = max(0.0, min(x2, float(w)) - max(x1, 0.0))
    iy = max(0.0, min(y2, float(h)) - max(y1, 0.0))
    return (ix * iy) / area


def _reconstruct_rigid_box(
    cb: Box,
    w_gold: float,
    h_gold: float,
    est_frame_w: Optional[float],
    est_frame_h: Optional[float] = None,
) -> Tuple[Box, bool]:
    """
    Reconstruct full rigid container geometry when screen-edge clipping collapses YOLO's bbox.
    Returns (rigid_box, is_unclipped_interior).
    """
    cw = cb[2] - cb[0]
    ch = cb[3] - cb[1]
    if w_gold <= 1.0:
        return cb, True

    ratio_w = cw / w_gold
    ratio_h = (ch / h_gold) if h_gold > 1.0 else 1.0
    x1, y1, x2, y2 = cb

    # Horizontal reconstruction anchored on whichever physical edge lies inside the frame
    if not (0.78 <= ratio_w <= 1.25):
        mid_x = (cb[0] + cb[2]) / 2.0
        half_w = (est_frame_w / 2.0) if est_frame_w is not None else 640.0
        on_left_half = mid_x < half_w and (est_frame_w is None or cb[2] < est_frame_w - 25.0)
        if on_left_half:
            x1, x2 = cb[2] - w_gold, cb[2]
        else:
            x1, x2 = cb[0], cb[0] + w_gold

    # Vertical reconstruction anchored on reliable top or bottom edge
    if h_gold > 1.0 and ratio_h < 0.78:
        mid_y = (cb[1] + cb[3]) / 2.0
        half_h = (est_frame_h / 2.0) if est_frame_h is not None else 540.0
        if cb[1] <= 40.0 or mid_y < half_h:
            y1, y2 = cb[3] - h_gold, cb[3]
        else:
            y1, y2 = cb[1], cb[1] + h_gold

    touches_edge = (
        cb[0] <= 18.0
        or (est_frame_w is not None and cb[2] >= est_frame_w - 18.0)
        or cb[1] <= 18.0
        or (est_frame_h is not None and cb[3] >= est_frame_h - 18.0)
    )
    is_interior = (0.78 <= ratio_w <= 1.25) and (0.78 <= ratio_h <= 1.25) and (not touches_edge)
    return (float(x1), float(y1), float(x2), float(y2)), is_interior


def _filter_num_anchors_by_golden(
    raw_anchors: Dict[int, Box],
    golden_num_box: Optional[Box],
    golden_f0: Optional[int],
    C: Optional[np.ndarray],
) -> Dict[int, Box]:
    """Reject number anchors that jumped to an adjacent container."""
    if C is None or len(C) == 0 or golden_num_box is None or golden_f0 is None:
        return dict(raw_anchors)
    cx_gold = (golden_num_box[0] + golden_num_box[2]) / 2.0
    cy_gold = (golden_num_box[1] + golden_num_box[3]) / 2.0
    gw = max(1.0, golden_num_box[2] - golden_num_box[0])
    gh = max(1.0, golden_num_box[3] - golden_num_box[1])
    valid: Dict[int, Box] = {}
    for a_f, a_box in raw_anchors.items():
        if not (0 <= a_f < len(C) and 0 <= golden_f0 < len(C)):
            continue
        aw = max(1.0, a_box[2] - a_box[0])
        ah = max(1.0, a_box[3] - a_box[1])
        # Size consistency guard against sudden merge/split anomalies
        if not (0.45 <= aw / gw <= 2.1 and 0.45 <= ah / gh <= 2.1):
            continue
        mapped_back = map_box(a_box, C, a_f, golden_f0)
        cx_map = (mapped_back[0] + mapped_back[2]) / 2.0
        cy_map = (mapped_back[1] + mapped_back[3]) / 2.0
        dist = float(np.hypot(cx_map - cx_gold, cy_map - cy_gold))
        tol = 80.0 + 0.60 * abs(a_f - golden_f0)
        if dist <= tol:
            valid[a_f] = a_box
    if not valid:
        valid[golden_f0] = golden_num_box
    return valid


def overlays_from_front_rear_tracks(
    tracks: Iterable[Any],
    label_fn: Callable[[Any], str],
    C: Optional[np.ndarray] = None,
) -> List[Overlay]:
    """
    Bangun Overlay dari _ContainerTrack untuk mode FRONT_REAR.
    Menghilangkan penyusutan kotak akibat edge clipping dan menolak anchor yang melompat ke tetangga.
    """
    track_list = list(tracks)
    max_x2 = 0.0
    max_y2 = 0.0
    for t in track_list:
        for cb_item in getattr(t, "history", {}).values():
            if cb_item is None:
                continue
            x2 = float(cb_item.x2) if hasattr(cb_item, "x2") else float(cb_item[2])
            y2 = float(cb_item.y2) if hasattr(cb_item, "y2") else float(cb_item[3])
            if x2 > max_x2:
                max_x2 = x2
            if y2 > max_y2:
                max_y2 = y2
    est_frame_w: Optional[float] = None
    est_frame_h: Optional[float] = None
    if max_x2 >= 2500.0:
        est_frame_w, est_frame_h = 3840.0, 2160.0
    elif max_x2 >= 1500.0:
        est_frame_w, est_frame_h = 1920.0, 1080.0

    out: List[Overlay] = []
    for t in track_list:
        raw_cont: Dict[int, Box] = {}
        for fidx, cb_item in getattr(t, "history", {}).items():
            if cb_item is None:
                continue
            if hasattr(cb_item, "x1") and hasattr(cb_item, "y1"):
                raw_cont[int(fidx) - 1] = (float(cb_item.x1), float(cb_item.y1), float(cb_item.x2), float(cb_item.y2))
            elif isinstance(cb_item, (list, tuple)) and len(cb_item) >= 4:
                raw_cont[int(fidx) - 1] = (float(cb_item[0]), float(cb_item[1]), float(cb_item[2]), float(cb_item[3]))

        raw_anchors: Dict[int, Box] = {}
        for fidx, b in getattr(t, "num_history", {}).items():
            raw_anchors[int(fidx) - 1] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

        nb = getattr(t, "best_nb", None)
        bf = getattr(t, "best_frame_idx", None)
        cb = getattr(t, "best_cb", None)

        golden_num_box: Optional[Box] = None
        golden_cont_box: Optional[Box] = None
        golden_f0: Optional[int] = (int(bf) - 1) if bf is not None else None

        if cb is not None:
            golden_cont_box = (float(cb.x1), float(cb.y1), float(cb.x2), float(cb.y2))
            if golden_f0 is not None and golden_f0 not in raw_cont:
                raw_cont[golden_f0] = golden_cont_box

        if nb is not None and golden_f0 is not None:
            golden_num_box = (float(nb.x1), float(nb.y1), float(nb.x2), float(nb.y2))
            raw_anchors[golden_f0] = golden_num_box
        elif cb is not None and golden_f0 is not None:
            golden_num_box = (
                float(cb.x1 + 0.45 * cb.width),
                float(cb.y1 + 0.12 * cb.height),
                float(cb.x1 + 0.95 * cb.width),
                float(cb.y1 + 0.38 * cb.height),
            )
            raw_anchors.setdefault(golden_f0, golden_num_box)

        if not raw_anchors and not raw_cont:
            continue

        # Filter container observations against golden box width so edge clipping never shrinks/slides boxes
        w_gold = (golden_cont_box[2] - golden_cont_box[0]) if golden_cont_box else 0.0
        h_gold = (golden_cont_box[3] - golden_cont_box[1]) if golden_cont_box else 0.0
        cont_hist_rigid: Dict[int, Box] = {}
        cont_unclipped: Dict[int, Box] = {}
        for cf, cbox in raw_cont.items():
            rigid_box, is_interior = _reconstruct_rigid_box(cbox, w_gold, h_gold, est_frame_w, est_frame_h)
            cont_hist_rigid[cf] = rigid_box
            if is_interior:
                cont_unclipped[cf] = cbox
        if golden_f0 is not None and golden_cont_box is not None:
            cont_unclipped.setdefault(golden_f0, golden_cont_box)

        rel_badge: Optional[Tuple[float, float, float, float]] = None
        if golden_cont_box is not None and golden_num_box is not None:
            cw = max(1.0, float(golden_cont_box[2] - golden_cont_box[0]))
            ch = max(1.0, float(golden_cont_box[3] - golden_cont_box[1]))
            rel_badge = (
                (golden_num_box[0] - golden_cont_box[0]) / cw,
                (golden_num_box[1] - golden_cont_box[1]) / ch,
                (golden_num_box[2] - golden_cont_box[0]) / cw,
                (golden_num_box[3] - golden_cont_box[1]) / ch,
            )

        valid_anchors = _filter_num_anchors_by_golden(raw_anchors, golden_num_box, golden_f0, C)

        # Normalize number anchor sizes to golden size if rigid geometry is known
        if golden_num_box is not None and valid_anchors:
            gnw = golden_num_box[2] - golden_num_box[0]
            gnh = golden_num_box[3] - golden_num_box[1]
            for af in list(valid_anchors.keys()):
                ax1, ay1, ax2, ay2 = valid_anchors[af]
                # Keep unclipped anchor center; pin size only if box was edge-clipped vertically/horizontally
                if abs((ax2 - ax1) - gnw) > 0.25 * gnw or abs((ay2 - ay1) - gnh) > 0.25 * gnh:
                    acx = (ax1 + ax2) / 2.0
                    acy = (ay1 + ay2) / 2.0
                    valid_anchors[af] = (acx - gnw / 2.0, acy - gnh / 2.0, acx + gnw / 2.0, acy + gnh / 2.0)

        first_f = min(cont_hist_rigid.keys()) if cont_hist_rigid else min(valid_anchors.keys())
        last_f = max(cont_hist_rigid.keys()) if cont_hist_rigid else max(valid_anchors.keys())

        out.append(
            Overlay(
                label=label_fn(t),
                is_valid=bool(getattr(t, "is_valid", False)),
                anchors=valid_anchors,
                first_frame=first_f,
                last_frame=last_f,
                golden_frame=golden_f0,
                golden_container_box=golden_cont_box,
                container_history=cont_hist_rigid,
                container_anchors=cont_unclipped,
                rel_badge=rel_badge,
                max_extrapolation=260,
                min_number_visible=0.50,
                min_container_visible=0.15,
                is_stitching=False,
            )
        )
    return out


def _project_box_homography(box_xywh: Tuple[float, float, float, float], H: np.ndarray) -> Optional[Box]:
    x, y, w, h = box_xywh
    if w <= 0 or h <= 0:
        return None
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H)
    if q is None or not np.isfinite(q).all():
        return None
    xy = q.reshape(-1, 2)
    bx1, by1, bx2, by2 = float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())
    if bx2 <= bx1 or by2 <= by1:
        return None
    return (bx1, by1, bx2, by2)


def overlays_from_stitching_tracks(
    tracks_data: Iterable[Dict[str, Any]],
    C: Optional[np.ndarray] = None,
) -> List[Overlay]:
    """
    Bangun Overlay dari data track stitching (TOP / LEFT_RIGHT).
    Menggunakan multi-anchor H_pano_to_vid dari frame_registrations & direct_4k_anchor
    agar kotak kontainer & nomor menempel presisi sepanjang lintasan masuk hingga keluar layar.
    """
    out: List[Overlay] = []
    for t in tracks_data:
        c_num = t.get("clean_text") or t.get("raw_text") or f"R{t.get('route_order', 1):03d}"
        iso_c = t.get("iso_text", "")
        disp = f"{c_num}" + (f" | {iso_c}" if iso_c else "")

        dets = t.get("detections", {})
        num_det = dets.get("container_number") if isinstance(dets, dict) else None
        nb = num_det.get("bbox") if isinstance(num_det, dict) else None
        iso_det = dets.get("iso_code") if isinstance(dets, dict) else None
        iso_b = iso_det.get("bbox") if isinstance(iso_det, dict) else None
        pano_cb = t.get("panorama_container_bbox")
        direct_4k = t.get("direct_4k_anchor")
        frame_regs = t.get("frame_registrations") or []

        num_anchors: Dict[int, Box] = {}
        cont_anchors: Dict[int, Box] = {}

        # Panorama number box (x, y, w, h) — prefer 4K-verified box back-projected onto panorama if available
        nb_xywh: Optional[Tuple[float, float, float, float]] = None
        if nb and isinstance(nb, dict) and float(nb.get("width", 0)) > 0 and float(nb.get("height", 0)) > 0:
            nb_xywh = (float(nb["x"]), float(nb["y"]), float(nb["width"]), float(nb["height"]))
        iso_pano_box: Optional[Box] = None
        if iso_b and isinstance(iso_b, dict) and float(iso_b.get("width", 0)) > 0 and float(iso_b.get("height", 0)) > 0:
            ix, iy, iw, ih = float(iso_b["x"]), float(iso_b["y"]), float(iso_b["width"]), float(iso_b["height"])
            iso_pano_box = (ix, iy, ix + iw, iy + ih)

        H_by_frame: Dict[int, np.ndarray] = {}
        det_by_frame: Dict[int, Box] = {}
        for reg in frame_regs:
            try:
                rf = int(reg["frame"])
                H = np.asarray(reg["H_pano_to_vid"], dtype=np.float64)
                db = reg.get("det_bbox")
                if H.shape == (3, 3) and np.isfinite(H).all():
                    H_by_frame[rf] = H
                if db and len(db) >= 4:
                    det_by_frame[rf] = (float(db[0]), float(db[1]), float(db[2]), float(db[3]))
            except Exception:
                continue

        # Track whether direct_4k agrees with Stage-4 detection and does not drift onto ISO code
        orig_pano_nb = nb_xywh
        direct_4k_synced = False
        if isinstance(direct_4k, dict) and direct_4k.get("num_box") is not None:
            f4k = int(direct_4k["frame"])
            n4k = tuple(float(v) for v in direct_4k["num_box"][:4])
            n4k_box: Box = (n4k[0], n4k[1], n4k[2], n4k[3])
            agrees_with_pano = True
            if f4k in H_by_frame:
                try:
                    H_inv = np.linalg.inv(H_by_frame[f4k])
                    pano_nb = _project_box_homography(
                        (n4k[0], n4k[1], n4k[2] - n4k[0], n4k[3] - n4k[1]), H_inv
                    )
                    if pano_nb is not None:
                        hits_iso = iso_pano_box is not None and box_iou(iso_pano_box, pano_nb) >= 0.20
                        if hits_iso and orig_pano_nb is not None:
                            agrees_with_pano = False
                        elif orig_pano_nb is not None:
                            orig_box = (
                                orig_pano_nb[0],
                                orig_pano_nb[1],
                                orig_pano_nb[0] + orig_pano_nb[2],
                                orig_pano_nb[1] + orig_pano_nb[3],
                            )
                            ocx = (orig_box[0] + orig_box[2]) / 2.0
                            pcx = (pano_nb[0] + pano_nb[2]) / 2.0
                            agrees_with_pano = box_iou(orig_box, pano_nb) >= 0.25 or abs(pcx - ocx) <= 95.0
                        if agrees_with_pano:
                            nb_xywh = (pano_nb[0], pano_nb[1], pano_nb[2] - pano_nb[0], pano_nb[3] - pano_nb[1])
                            direct_4k_synced = True
                except np.linalg.LinAlgError:
                    pass
            if agrees_with_pano or orig_pano_nb is None:
                num_anchors[f4k] = n4k_box

        # Panorama container geometry for unclipped multi-frame projection across edges
        pano_cb_xywh: Optional[Tuple[float, float, float, float]] = None
        if isinstance(pano_cb, (list, tuple)) and len(pano_cb) >= 4:
            px1, py1, px2, py2 = float(pano_cb[0]), float(pano_cb[1]), float(pano_cb[2]), float(pano_cb[3])
            if px2 > px1 and py2 > py1:
                pano_cb_xywh = (px1, py1, px2 - px1, py2 - py1)
        elif isinstance(pano_cb, dict) and float(pano_cb.get("width", 0)) > 0:
            pano_cb_xywh = (
                float(pano_cb["x"]), float(pano_cb["y"]), float(pano_cb["width"]), float(pano_cb["height"])
            )

        # Infer video frame bounds & typical projected dimensions to snap unclipped sides & prevent projective blowup
        max_db_x = max((float(b[2]) for b in det_by_frame.values()), default=0.0)
        max_db_y = max((float(b[3]) for b in det_by_frame.values()), default=0.0)
        vid_w_est = max(3840.0, max_db_x) if max_db_x > 1950.0 else max(1920.0, max_db_x)
        vid_h_est = max(2160.0, max_db_y) if max_db_y > 1100.0 else max(1080.0, max_db_y)
        raw_proj_cbs: Dict[int, Box] = {}
        if pano_cb_xywh is not None:
            for rf, H in H_by_frame.items():
                cand_pcb = _project_box_homography(pano_cb_xywh, H)
                if cand_pcb is not None and 60.0 <= (cand_pcb[2] - cand_pcb[0]) <= 7000.0 and 60.0 <= (cand_pcb[3] - cand_pcb[1]) <= 3200.0:
                    raw_proj_cbs[rf] = cand_pcb
        med_pw = float(np.median([b[2] - b[0] for b in raw_proj_cbs.values()])) if raw_proj_cbs else None
        med_ph = float(np.median([b[3] - b[1] for b in raw_proj_cbs.values()])) if raw_proj_cbs else None

        # If direct_4k succeeded on an unregistered frame, do not project unverified panorama nb_xywh
        allow_pano_num_proj = not (isinstance(direct_4k, dict) and direct_4k.get("num_box") is not None and not direct_4k_synced and orig_pano_nb is None)

        for rf in sorted(set(H_by_frame.keys()) | set(det_by_frame.keys())):
            db = det_by_frame.get(rf)
            H = H_by_frame.get(rf)
            proj_cb = raw_proj_cbs.get(rf)

            # Snap ONLY genuinely unclipped borders to db; keep proj_cb on fragments extending off-screen
            cb_rf: Optional[Box] = None
            if proj_cb is not None and db is not None:
                full_db_horiz = db[0] > 80.0 and db[2] < (vid_w_est - 80.0) and (med_pw is None or (db[2] - db[0]) >= 0.72 * med_pw)
                l_free = db[0] > 40.0 and (proj_cb[0] >= -90.0 or full_db_horiz)
                r_free = db[2] < (vid_w_est - 40.0) and (proj_cb[2] <= (vid_w_est + 90.0) or full_db_horiz)
                t_free = db[1] > 28.0 and (proj_cb[1] >= -90.0 or (med_ph is not None and (db[3] - db[1]) >= 0.72 * med_ph))
                b_free = db[3] < (vid_h_est - 28.0) and (proj_cb[3] <= (vid_h_est + 90.0) or (med_ph is not None and (db[3] - db[1]) >= 0.72 * med_ph))
                x1 = db[0] if l_free else proj_cb[0]
                y1 = db[1] if t_free else proj_cb[1]
                x2 = db[2] if r_free else proj_cb[2]
                y2 = db[3] if b_free else proj_cb[3]
                if med_pw is not None:
                    max_w = 1.25 * med_pw
                    if not l_free and r_free and (x2 - x1) > max_w:
                        x1 = x2 - max_w
                    elif not r_free and l_free and (x2 - x1) > max_w:
                        x2 = x1 + max_w
                if med_ph is not None:
                    max_h = 1.18 * med_ph
                    if not t_free and b_free and (y2 - y1) > max_h:
                        y1 = y2 - max_h
                    elif not b_free and t_free and (y2 - y1) > max_h:
                        y2 = y1 + max_h
                if x2 > x1 + 20.0 and y2 > y1 + 20.0:
                    cb_rf = (float(x1), float(y1), float(x2), float(y2))
                else:
                    cb_rf = proj_cb
            else:
                cb_rf = proj_cb if proj_cb is not None else db

            if cb_rf is not None and rf not in cont_anchors:
                cont_anchors[rf] = cb_rf

            # Project verified number box onto all registered frames (checking against unclipped proj_cb, NOT clipped det_bbox)
            if allow_pano_num_proj and H is not None and nb_xywh is not None and rf not in num_anchors:
                proj_nb = _project_box_homography(nb_xywh, H)
                ref_cb = cb_rf if cb_rf is not None else proj_cb
                if proj_nb is not None:
                    nw = proj_nb[2] - proj_nb[0]
                    nh = proj_nb[3] - proj_nb[1]
                    ncx = (proj_nb[0] + proj_nb[2]) / 2.0
                    ncy = (proj_nb[1] + proj_nb[3]) / 2.0
                    sane_size = (8.0 <= nw <= 1800.0) and (8.0 <= nh <= 1800.0)
                    inside_cont = True
                    if ref_cb is not None:
                        inside_cont = (
                            (ref_cb[0] - 140.0 <= ncx <= ref_cb[2] + 140.0)
                            and (ref_cb[1] - 140.0 <= ncy <= ref_cb[3] + 140.0)
                        )
                    if sane_size and inside_cont:
                        num_anchors[rf] = proj_nb

        # Only fill direct_4k container box if frame wasn't already covered by snapped/unclipped frame_registrations
        if isinstance(direct_4k, dict) and direct_4k.get("cont_box") is not None:
            f4k_c = int(direct_4k["frame"])
            if f4k_c not in cont_anchors:
                c4k = tuple(float(v) for v in direct_4k["cont_box"][:4])
                cont_anchors[f4k_c] = (c4k[0], c4k[1], c4k[2], c4k[3])

        # Legacy single-frame fallback when frame_registrations is absent
        pf = t.get("pass_frame")
        vbox = t.get("container_video_bbox")
        golden_f: Optional[int] = None
        golden_cont_box: Optional[Box] = None

        if cont_anchors:
            # Prefer center-most unclipped frame as golden frame
            golden_f = max(
                cont_anchors.keys(),
                key=lambda fk: (cont_anchors[fk][2] - cont_anchors[fk][0]) * (1.0 if fk in num_anchors else 0.5),
            )
            golden_cont_box = cont_anchors[golden_f]
        elif pf is not None and vbox and isinstance(vbox, (list, tuple)) and len(vbox) >= 4:
            golden_f = max(0, int(pf))
            vx1, vy1, vx2, vy2 = float(vbox[0]), float(vbox[1]), float(vbox[2]), float(vbox[3])
            golden_cont_box = (vx1, vy1, vx2, vy2)
            cont_anchors[golden_f] = golden_cont_box

            if nb_xywh is not None and not num_anchors:
                rw = float(t.get("pano_w") or 0.0)
                rh = float(t.get("pano_h") or 0.0)
                if rw <= 0.0 or rh <= 0.0:
                    rw, rh = 1318.0, 320.0
                    img_p = t.get("image_path")
                    if img_p and Path(str(img_p)).is_file():
                        im = cv2.imread(str(img_p))
                        if im is not None:
                            rh, rw = float(im.shape[0]), float(im.shape[1])
                vw = max(1.0, vx2 - vx1)
                vh = max(1.0, vy2 - vy1)
                sx, sy = vw / rw, vh / rh
                tx1 = vx1 + nb_xywh[0] * sx
                ty1 = vy1 + nb_xywh[1] * sy
                num_anchors[golden_f] = (tx1, ty1, tx1 + nb_xywh[2] * sx, ty1 + nb_xywh[3] * sy)

        if not num_anchors and not cont_anchors:
            continue

        if not num_anchors and golden_f is not None and golden_cont_box is not None:
            vx1, vy1, vx2, vy2 = golden_cont_box
            cx = (vx1 + vx2) / 2.0
            cy = vy1 + (vy2 - vy1) * 0.25
            num_anchors[golden_f] = (cx - 140.0, cy - 35.0, cx + 140.0, cy + 35.0)

        # Reject any stray projected num_anchors that contradict the median relative offset
        if len(num_anchors) >= 3 and cont_anchors:
            shared = [fk for fk in num_anchors if fk in cont_anchors]
            if len(shared) >= 2:
                rel_cxs = [
                    ((num_anchors[fk][0] + num_anchors[fk][2]) / 2.0 - cont_anchors[fk][0])
                    / max(1.0, cont_anchors[fk][2] - cont_anchors[fk][0])
                    for fk in shared
                ]
                med_rx = float(np.median(rel_cxs))
                filtered_num: Dict[int, Box] = {}
                for fk, nbox in num_anchors.items():
                    if fk in cont_anchors:
                        cbk = cont_anchors[fk]
                        rx = ((nbox[0] + nbox[2]) / 2.0 - cbk[0]) / max(1.0, cbk[2] - cbk[0])
                        if abs(rx - med_rx) <= 0.25:
                            filtered_num[fk] = nbox
                    else:
                        filtered_num[fk] = nbox
                if filtered_num:
                    num_anchors = filtered_num

        if C is not None and len(C) > 0 and golden_cont_box is None and cont_anchors:
            golden_f = sorted(cont_anchors.keys())[len(cont_anchors) // 2]
            golden_cont_box = cont_anchors[golden_f]

        all_frames = sorted(set(cont_anchors.keys()) | set(num_anchors.keys()))
        first_f = all_frames[0]
        last_f = all_frames[-1]

        out.append(
            Overlay(
                label=disp,
                is_valid=bool(t.get("is_valid", False)),
                anchors=num_anchors,
                first_frame=first_f,
                last_frame=last_f,
                golden_frame=golden_f,
                golden_container_box=golden_cont_box,
                container_anchors=cont_anchors,
                max_extrapolation=150,
                min_number_visible=0.35,
                min_container_visible=0.10,
                is_stitching=True,
            )
        )
    return out


# --------------------------------------------------------------------------
# 3. Single-pass rolling-buffer ROI-Local Forward-Backward LK Tracker
# --------------------------------------------------------------------------
def _roi_step(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    box: Box,
    w0: int,
    h0: int,
    ww: int,
    hs: int,
    last_v: Tuple[float, float],
) -> Tuple[float, float]:
    """Compute container translation (dx, dy) in full-res pixels using inset ROI FB-LK."""
    sx = ww / float(w0)
    sy = hs / float(h0)
    bx1 = max(0.0, box[0])
    by1 = max(0.0, box[1])
    bx2 = min(float(w0), box[2])
    by2 = min(float(h0), box[3])
    bw = bx2 - bx1
    bh = by2 - by1
    if bw * sx < 24.0 or bh * sy < 24.0:
        return last_v[0] * 0.7, last_v[1] * 0.7

    # Inset by 8% so neighbor container borders don't pollute optical flow
    ix1 = int(round((bx1 + 0.08 * bw) * sx))
    iy1 = int(round((by1 + 0.08 * bh) * sy))
    ix2 = int(round((bx2 - 0.08 * bw) * sx))
    iy2 = int(round((by2 - 0.08 * bh) * sy))
    if ix2 - ix1 < 20 or iy2 - iy1 < 20:
        ix1 = max(0, int(round(bx1 * sx)))
        iy1 = max(0, int(round(by1 * sy)))
        ix2 = min(ww, int(round(bx2 * sx)))
        iy2 = min(hs, int(round(by2 * sy)))
    if ix2 - ix1 < 16 or iy2 - iy1 < 16:
        return last_v[0] * 0.7, last_v[1] * 0.7

    mask = np.zeros((hs, ww), dtype=np.uint8)
    mask[max(0, iy1):min(hs, iy2), max(0, ix1):min(ww, ix2)] = 255
    d = _lk_translation(
        prev_gray, cur_gray, mask=mask, max_corners=360, win_size=25, quality_level=0.008, min_distance=5
    )
    if d is not None:
        return d[0] / sx, d[1] / sy
    return last_v[0] * 0.8, last_v[1] * 0.8


def _roi_pair_step(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    cb: Box,
    nb: Box,
    w0: int,
    h0: int,
    ww: int,
    hs: int,
    last_v: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Estimate (dx, dy) for container and a tight local text-corner LK refinement for number box."""
    dc = _roi_step(prev_gray, cur_gray, cb, w0, h0, ww, hs, last_v)
    if _visible_fraction(nb, w0, h0) < 0.35:
        return dc, dc

    sx = ww / float(w0)
    sy = hs / float(h0)
    nw = max(1.0, nb[2] - nb[0])
    nh = max(1.0, nb[3] - nb[1])
    bw = max(1.0, cb[2] - cb[0])
    bh = max(1.0, cb[3] - cb[1])

    # Search textured zone around number/ISO/corner while strictly clamped inside container bounds
    pad_x = max(160.0, 2.5 * nw)
    pad_y = max(90.0, 0.35 * nh)
    rx1 = max(0.0, max(cb[0] + 0.02 * bw, nb[0] - pad_x))
    ry1 = max(0.0, max(cb[1] + 0.02 * bh, nb[1] - pad_y))
    rx2 = min(float(w0), min(cb[2] - 0.02 * bw, nb[2] + pad_x))
    ry2 = min(float(h0), min(cb[3] - 0.02 * bh, nb[3] + pad_y))
    mx1, my1 = int(round(rx1 * sx)), int(round(ry1 * sy))
    mx2, my2 = int(round(rx2 * sx)), int(round(ry2 * sy))
    if (mx2 - mx1) >= 24 and (my2 - my1) >= 24:
        mask = np.zeros((hs, ww), dtype=np.uint8)
        mask[max(0, my1):min(hs, my2), max(0, mx1):min(ww, mx2)] = 255
        dn = _lk_translation(
            prev_gray, cur_gray, mask=mask, max_corners=180, win_size=25, quality_level=0.008, min_distance=4
        )
        if dn is not None:
            ndx, ndy = dn[0] / sx, dn[1] / sy
            if abs(ndx - dc[0]) <= 28.0 and abs(ndy - dc[1]) <= 28.0:
                return dc, (ndx, ndy)
    return dc, dc


def _shift_box(b: Box, dx: float, dy: float) -> Box:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _compute_roi_lk_tracks(
    video_path: str,
    overlays: List[Overlay],
    w0: int,
    h0: int,
    total_frames: int,
    work_width: int = 960,
    max_edge_extend: int = 260,
    frame_stride: int = 2,
    cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    """
    Populate _resolved_cont and _resolved_num on each Overlay with ROI-local FB-LK tracking:
      - Forward piecewise between reliable anchors (with linear residual blending)
      - Forward off the exit edge until container visibility < min_container_visible
      - Backward over a rolling 960px grayscale buffer from first anchor to screen entry edge
    """
    if not overlays:
        return

    hs = max(2, int(round(work_width * h0 / float(max(1, w0)))))

    # Build per-overlay sorted anchor map
    ov_states: List[Dict[str, Any]] = []
    for ov in overlays:
        ov._resolved_cont = {}
        ov._resolved_num = {}

        cont_src: Dict[int, Box] = dict(ov.container_anchors) if ov.container_anchors else dict(ov.container_history)
        if not cont_src and ov.golden_frame is not None and ov.golden_container_box is not None:
            cont_src[ov.golden_frame] = ov.golden_container_box
        num_src: Dict[int, Box] = dict(ov.anchors)

        cont_keys = sorted(cont_src.keys())
        num_keys = sorted(num_src.keys())
        if not cont_keys and not num_keys:
            continue

        # Reference rigid sizes (median / golden) for pairing cont & num when one anchor is missing
        if ov.golden_container_box is not None:
            ref_cb = ov.golden_container_box
            ref_cf = ov.golden_frame if ov.golden_frame is not None else (cont_keys[0] if cont_keys else num_keys[0])
        elif cont_keys:
            ref_cf = cont_keys[len(cont_keys) // 2]
            ref_cb = cont_src[ref_cf]
        else:
            ref_cf = num_keys[0]
            ref_cb = num_src[ref_cf]

        if ov.golden_frame is not None and ov.golden_frame in num_src:
            ref_nf = ov.golden_frame
            ref_nb = num_src[ref_nf]
        elif num_keys:
            ref_nf = min(num_keys, key=lambda fk: abs(fk - ref_cf))
            ref_nb = num_src[ref_nf]
        else:
            ref_nf = ref_cf
            ref_nb = (
                ref_cb[0] + 0.55 * (ref_cb[2] - ref_cb[0]),
                ref_cb[1] + 0.15 * (ref_cb[3] - ref_cb[1]),
                ref_cb[0] + 0.92 * (ref_cb[2] - ref_cb[0]),
                ref_cb[1] + 0.42 * (ref_cb[3] - ref_cb[1]),
            )

        # Rigid vector from cont center to num center/box when single source misses frames
        ref_cw = max(1.0, ref_cb[2] - ref_cb[0])
        ref_ch = max(1.0, ref_cb[3] - ref_cb[1])
        nw = ref_nb[2] - ref_nb[0]
        nh = ref_nb[3] - ref_nb[1]

        first_f = min(cont_keys[0] if cont_keys else num_keys[0], num_keys[0] if num_keys else cont_keys[0])
        last_f = max(cont_keys[-1] if cont_keys else num_keys[-1], num_keys[-1] if num_keys else cont_keys[-1])
        ov_states.append({
            "ov": ov,
            "cont_src": cont_src,
            "num_src": num_src,
            "cont_keys": cont_keys,
            "num_keys": num_keys,
            "first_anchor_f": first_f,
            "last_anchor_f": last_f,
            "ref_cb": ref_cb,
            "ref_nb": ref_nb,
            "rel_off": (
                (ref_nb[0] - ref_cb[0]) / ref_cw,
                (ref_nb[1] - ref_cb[1]) / ref_ch,
                (ref_nb[2] - ref_cb[0]) / ref_cw,
                (ref_nb[3] - ref_cb[1]) / ref_ch,
            ) if ov.golden_container_box is not None or cont_keys else None,
            "nw": nw,
            "nh": nh,
            "cur_cb": None,
            "cur_nb": None,
            "last_v": (0.0, 0.0),
            "raw_fwd_cb": {},
            "raw_fwd_nb": {},
            "back_done": False,
            "fwd_done": False,
        })

    if not ov_states:
        return

    min_active_f = max(0, min(s["first_anchor_f"] for s in ov_states) - max_edge_extend)
    max_active_f = min(total_frames - 1 if total_frames > 0 else 10**9, max(s["last_anchor_f"] for s in ov_states) + max_edge_extend)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    gray_ring: deque[Tuple[int, np.ndarray]] = deque(maxlen=max_edge_extend + 5)
    prev_gray: Optional[np.ndarray] = None
    f = 0

    try:
        while True:
            if cancel is not None and cancel():
                break
            if f > max_active_f:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if f >= min_active_f - max_edge_extend:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if g.shape[1] != work_width:
                    g = cv2.resize(g, (work_width, hs), interpolation=cv2.INTER_AREA)
            else:
                g = None  # skip resize completely outside active windows

            if g is not None:
                # 1. Seed or propagate forward
                for s in ov_states:
                    ov: Overlay = s["ov"]
                    fa = s["first_anchor_f"]
                    la = s["last_anchor_f"]

                    # Propagate from previous frame f-1 -> f if active
                    if s["cur_cb"] is not None and not s["fwd_done"] and prev_gray is not None:
                        if (f % frame_stride) == 0:
                            # LK penuh (forward-backward check)
                            (dx, dy), (ndx, ndy) = _roi_pair_step(
                                prev_gray, g, s["cur_cb"], s["cur_nb"], w0, h0, work_width, hs, s["last_v"]
                            )
                            s["last_v"] = (dx, dy)
                        else:
                            # Frame di-skip: pakai velocity dari frame sebelumnya (containers move smoothly)
                            dx, dy = s["last_v"]
                            ndx, ndy = s["last_v"]
                        pred_cb = _shift_box(s["cur_cb"], dx, dy)
                        pred_nb = _shift_box(s["cur_nb"], ndx, ndy)
                        s["raw_fwd_cb"][f] = pred_cb
                        s["raw_fwd_nb"][f] = pred_nb

                        # If f is an anchor frame, reset running state to exact anchor next
                        nxt_cb = s["cont_src"].get(f)
                        nxt_nb = s["num_src"].get(f)
                        if nxt_cb is not None:
                            s["cur_cb"] = nxt_cb
                        else:
                            s["cur_cb"] = pred_cb
                        if nxt_nb is not None:
                            s["cur_nb"] = nxt_nb
                        else:
                            s["cur_nb"] = pred_nb

                        # Stop extending past exit edge once container fades out
                        if f > la and (not Overlay._box_has_screen_presence(s["cur_cb"], w0, h0, 40.0) or f > la + max_edge_extend):
                            s["fwd_done"] = True

                    # Trigger initialization and backward entry tracking at first_anchor_f
                    if f == fa:
                        init_cb = s["cont_src"].get(fa)
                        init_nb = s["num_src"].get(fa)
                        if init_cb is None and init_nb is not None:
                            dx_n = (init_nb[0] + init_nb[2]) / 2.0 - (s["ref_nb"][0] + s["ref_nb"][2]) / 2.0
                            dy_n = (init_nb[1] + init_nb[3]) / 2.0 - (s["ref_nb"][1] + s["ref_nb"][3]) / 2.0
                            init_cb = _shift_box(s["ref_cb"], dx_n, dy_n)
                        if init_nb is None and init_cb is not None:
                            if s["num_keys"]:
                                nf0 = s["num_keys"][0]
                                nb0 = s["num_src"][nf0]
                                cb0 = s["cont_src"].get(nf0) or s["ref_cb"]
                                ncx0 = (nb0[0] + nb0[2]) / 2.0
                                ncy0 = (nb0[1] + nb0[3]) / 2.0
                                ccx0 = (cb0[0] + cb0[2]) / 2.0
                                ccy0 = (cb0[1] + cb0[3]) / 2.0
                                dx_e = (init_cb[2] - cb0[2]) if ncx0 >= ccx0 else (init_cb[0] - cb0[0])
                                dy_e = (init_cb[3] - cb0[3]) if ncy0 >= ccy0 else (init_cb[1] - cb0[1])
                                init_nb = _shift_box(nb0, dx_e, dy_e)
                            elif s["rel_off"] is not None:
                                cw0 = max(1.0, init_cb[2] - init_cb[0])
                                ch0 = max(1.0, init_cb[3] - init_cb[1])
                                rx1, ry1, rx2, ry2 = s["rel_off"]
                                if cw0 >= 0.50 * max(1.0, s["ref_cb"][2] - s["ref_cb"][0]):
                                    init_nb = (init_cb[0] + rx1 * cw0, init_cb[1] + ry1 * ch0, init_cb[0] + rx2 * cw0, init_cb[1] + ry2 * ch0)
                                else:
                                    ncx0 = init_cb[0] + 0.5 * (rx1 + rx2) * cw0
                                    ncy0 = init_cb[1] + 0.5 * (ry1 + ry2) * ch0
                                    hw = max(6.0, 0.5 * s["nw"])
                                    hh = max(6.0, 0.5 * s["nh"])
                                    init_nb = (ncx0 - hw, ncy0 - hh, ncx0 + hw, ncy0 + hh)
                        if init_cb is not None and init_nb is not None:
                            s["cur_cb"] = init_cb
                            s["cur_nb"] = init_nb
                            s["raw_fwd_cb"][fa] = init_cb
                            s["raw_fwd_nb"][fa] = init_nb

                            # Walk backwards across rolling grayscale ring buffer to entry edge
                            b_cb = init_cb
                            b_nb = init_nb
                            b_last_v = (0.0, 0.0)
                            cur_bg = g
                            cur_bf = fa
                            for rf, rg in reversed(gray_ring):
                                if cur_bf - rf > max_edge_extend:
                                    break
                                if rf >= fa:
                                    continue
                                if (cur_bf - rf) % frame_stride != 0:
                                    # skip LK, hanya shift dengan last velocity
                                    b_cb = _shift_box(b_cb, b_last_v[0], b_last_v[1])
                                    b_nb = _shift_box(b_nb, b_last_v[0], b_last_v[1])
                                    ov._resolved_cont[rf] = b_cb
                                    ov._resolved_num[rf] = b_nb
                                (dx, dy), (ndx, ndy) = _roi_pair_step(
                                    cur_bg, rg, b_cb, b_nb, w0, h0, work_width, hs, b_last_v
                                )
                                b_last_v = (dx, dy)
                                b_cb = _shift_box(b_cb, dx, dy)
                                b_nb = _shift_box(b_nb, ndx, ndy)
                                ov._resolved_cont[rf] = b_cb
                                ov._resolved_num[rf] = b_nb
                                cur_bg = rg
                                cur_bf = rf
                                if not Overlay._box_has_screen_presence(b_cb, w0, h0, 40.0):
                                    break
                            s["back_done"] = True

                gray_ring.append((f, g))
                prev_gray = g
            f += 1
            if progress_callback and total_frames > 0 and f % 40 == 0:
                progress_callback(min(1.0, f / float(total_frames)), f"ROI tracking presisi ({f}/{total_frames})...")
    finally:
        cap.release()

    # Finalize piecewise residual blending between consecutive anchors + tail forward
    for s in ov_states:
        ov = s["ov"]
        cont_src = s["cont_src"]
        num_src = s["num_src"]
        raw_cb = s["raw_fwd_cb"]
        raw_nb = s["raw_fwd_nb"]
        all_fwd = sorted(raw_cb.keys())
        if not all_fwd:
            continue

        # Blend piecewise container trajectory between consecutive cont_anchors
        ckeys = sorted(k for k in cont_src.keys() if k in raw_cb)
        _blend_piecewise(ov._resolved_cont, raw_cb, cont_src, ckeys, all_fwd)

        # Blend piecewise number trajectory between consecutive num_anchors
        nkeys = sorted(k for k in num_src.keys() if k in raw_nb)
        _blend_piecewise(ov._resolved_num, raw_nb, num_src, nkeys, all_fwd)


def _blend_piecewise(
    out_dict: Dict[int, Box],
    raw_fwd: Dict[int, Box],
    anchors: Dict[int, Box],
    anchor_keys: List[int],
    all_frames: List[int],
) -> None:
    """Apply linear residual correction between consecutive anchors so LK drift never steps at anchors."""
    if not anchor_keys:
        for fk in all_frames:
            if fk in raw_fwd:
                out_dict[fk] = raw_fwd[fk]
        return

    for idx_k, a0 in enumerate(anchor_keys):
        out_dict[a0] = anchors[a0]
        if idx_k + 1 < len(anchor_keys):
            a1 = anchor_keys[idx_k + 1]
            pred_a1 = raw_fwd.get(a1)
            true_a1 = anchors[a1]
            span = max(1, a1 - a0)
            res = (
                tuple(true_a1[j] - pred_a1[j] for j in range(4))
                if pred_a1 is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            for fk in range(a0 + 1, a1):
                rb = raw_fwd.get(fk)
                if rb is None:
                    continue
                alpha = (fk - a0) / float(span)
                out_dict[fk] = tuple(rb[j] + alpha * res[j] for j in range(4))  # type: ignore[assignment]

    # Pre-first-anchor and post-last-anchor frames from raw_fwd
    first_a = anchor_keys[0]
    last_a = anchor_keys[-1]
    for fk in all_frames:
        if (fk < first_a or fk > last_a) and fk in raw_fwd:
            out_dict[fk] = raw_fwd[fk]


# --------------------------------------------------------------------------
# 4. Render Video
# --------------------------------------------------------------------------
def render_overlay_video(
    video_path: str,
    out_path: str,
    overlays: List[Overlay],
    C: Optional[np.ndarray] = None,
    target_h: Optional[int] = None,
    cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    roi_work_width: int = 640,          # NEW — turun dari 960 default
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.release()

    if overlays:
        _compute_roi_lk_tracks(
            video_path=str(video_path),
            overlays=overlays,
            w0=w0, h0=h0, total_frames=total_frames,
            work_width=roi_work_width,        # ← passing param baru
            cancel=cancel,
            progress_callback=(
                (lambda pct, msg: progress_callback(pct * 0.35, msg))
                if progress_callback else None
            ),
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    if target_h and target_h != h0:
        tw = int(round(target_h * w0 / float(h0)))
        tw += tw % 2
        th = target_h
    else:
        tw, th = w0, h0
    sx, sy = tw / float(w0), th / float(h0)
    sf = max(0.8, th / 720.0)
    line = max(1, int(1.8 * sf))
    fscale, fthick = 0.42 * sf, (1 if sf < 1.4 else 2)
    color_container = (50, 205, 50)
    color_badge = (255, 255, 0)

    # Pre-render semua banner label (dipakai setiap frame)
    banner_cache = _build_banner_cache(overlays, sf, fscale, fthick, color_container)

    writer = FFmpegWriter(str(out_path), tw, th, fps, crf=23)
    if not writer.isOpened():
        cap.release()
        return False

    f = 0
    try:
        while True:
            if cancel is not None and cancel():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (tw, th) != (w0, h0):
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            for ov in overlays:
                cb = ov.container_box_at(f, C, w0, h0)
                b = ov.box_at(f, C, w0, h0)

                # 1. Kotak hijau + banner label (dari cache, tanpa putText)
                if cb is not None:
                    cx1 = int(round(cb[0] * sx)); cy1 = int(round(cb[1] * sy))
                    cx2 = int(round(cb[2] * sx)); cy2 = int(round(cb[3] * sy))
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color_container, line)

                    sprite = banner_cache.get(ov.label)
                    if sprite is not None:
                        sh, sw = sprite.shape[:2]
                        tx1 = max(0, min(tw - sw, cx1))
                        by1 = max(0, cy1 - sh)
                        tx2 = min(tw, tx1 + sw)
                        by2 = min(th, by1 + sh)
                        if by2 > by1 and tx2 > tx1:
                            frame[by1:by2, tx1:tx2] = sprite[:by2 - by1, :tx2 - tx1]

                # 2. Kotak cyan nomor kontainer
                if b is not None and (b[2] - b[0]) >= 6.0 and (b[3] - b[1]) >= 6.0:
                    bx1 = int(round(b[0] * sx)); by1 = int(round(b[1] * sy))
                    bx2 = int(round(b[2] * sx)); by2 = int(round(b[3] * sy))
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_badge, line)

                    # Banner fallback kalau kontainer tidak terdeteksi (tapi nomor ada)
                    if cb is None:
                        sprite = banner_cache.get(ov.label)
                        if sprite is not None:
                            sh, sw = sprite.shape[:2]
                            tx1 = max(0, min(tw - sw, bx1))
                            ly1 = max(0, by1 - sh)
                            tx2 = min(tw, tx1 + sw)
                            ly2 = min(th, ly1 + sh)
                            if ly2 > ly1 and tx2 > tx1:
                                frame[ly1:ly2, tx1:tx2] = sprite[:ly2 - ly1, :tx2 - tx1]

            writer.write(frame)
            f += 1
            if progress_callback and f % 10 == 0:
                base_p = 0.35 if overlays else 0.0
                scale_p = 0.65 if overlays else 1.0
                progress_callback(
                    base_p + scale_p * min(1.0, f / float(max(1, total_frames))),
                    f"Render video anotasi ({f}/{total_frames})...",
                )
    finally:
        cap.release()
        try:
            writer.release()
        except Exception as e:
            print(f"[VIDEO EXPORT] ffmpeg finalize error: {e}")

    return True
