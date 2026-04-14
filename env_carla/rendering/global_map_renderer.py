"""
Global map renderer for debug visualisation using OpenCV.

Displays the full road network, planned route, ego vehicle trace, and
distance travelled in a standalone window.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

Point2D = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


def world_to_gm_px(x, y, bounds, w, h, margin, flip_x=True, flip_y=False):
    """Transform world coordinates to global map pixel coordinates."""
    minx, maxx, miny, maxy = bounds
    sx = (x - minx) / max(maxx - minx, 1e-6)
    sy = (y - miny) / max(maxy - miny, 1e-6)
    sx = float(np.clip(sx, 0.0, 1.0))
    sy = float(np.clip(sy, 0.0, 1.0))
    if flip_x:
        sx = 1.0 - sx
    if flip_y:
        sy = 1.0 - sy
    px = int(margin + sx * (w - 1 - 2 * margin))
    py = int(margin + (1.0 - sy) * (h - 1 - 2 * margin))
    return px, py


@dataclass(frozen=True)
class GlobalMapConfig:
    window_name: str = "CARLA Global Map"
    win_size: Tuple[int, int] = (520, 520)
    margin_px: int = 10
    trace_len: int = 50000
    trace_draw: int = 5000
    draw_route: bool = True
    draw_trace: bool = True
    draw_ego: bool = True
    draw_text: bool = True
    flip_x: bool = True
    flip_y: bool = False


class GlobalMapRenderer:
    """OpenCV-based global map visualiser for debugging."""

    def __init__(self, config, gm_bg, bounds):
        self.cfg = config
        self.bounds = bounds
        self._w = int(self.cfg.win_size[0])
        self._h = int(self.cfg.win_size[1])
        self._bg = self._ensure_bg(gm_bg)
        self._ready = True
        self._route_plan_xy = []
        self._ego_trace_xy = []

        try:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cfg.window_name, self._w, self._h)
            cv2.imshow(self.cfg.window_name, self._bg)
            cv2.waitKey(1)
        except Exception:
            self._ready = False

    def _ensure_bg(self, bg):
        if bg is None:
            out = np.zeros((self._h, self._w, 3), dtype=np.uint8)
            out[:] = (10, 10, 10)
            return out
        if bg.shape[0] != self._h or bg.shape[1] != self._w:
            return cv2.resize(bg, (self._w, self._h), interpolation=cv2.INTER_AREA)
        return bg.copy()

    def is_ready(self):
        return bool(self._ready)

    def set_route_plan(self, route_plan_xy):
        self._route_plan_xy = list(route_plan_xy) if route_plan_xy else []

    def reset_trace(self):
        self._ego_trace_xy = []

    def update(self, ego_xy, total_distance_m=0.0):
        """Update the display with the current ego position."""
        if not self._ready:
            return

        ex, ey = float(ego_xy[0]), float(ego_xy[1])
        self._ego_trace_xy.append((ex, ey))
        if len(self._ego_trace_xy) > int(self.cfg.trace_len):
            self._ego_trace_xy = self._ego_trace_xy[-int(self.cfg.trace_len):]

        frame = self._bg.copy()

        if self.cfg.draw_route and len(self._route_plan_xy) >= 2:
            pts = [
                world_to_gm_px(x, y, self.bounds, self._w, self._h,
                               int(self.cfg.margin_px), flip_x=self.cfg.flip_x, flip_y=self.cfg.flip_y)
                for (x, y) in self._route_plan_xy
            ]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], (255, 255, 0), 2, cv2.LINE_AA)

        if self.cfg.draw_trace:
            pts = self._ego_trace_xy[-max(2, int(self.cfg.trace_draw)):]
            pix = [
                world_to_gm_px(x, y, self.bounds, self._w, self._h,
                               int(self.cfg.margin_px), flip_x=self.cfg.flip_x, flip_y=self.cfg.flip_y)
                for (x, y) in pts
            ]
            for i in range(len(pix) - 1):
                cv2.line(frame, pix[i], pix[i + 1], (0, 0, 255), 2, cv2.LINE_AA)

        if self.cfg.draw_ego:
            p = world_to_gm_px(ex, ey, self.bounds, self._w, self._h,
                               int(self.cfg.margin_px), flip_x=self.cfg.flip_x, flip_y=self.cfg.flip_y)
            cv2.circle(frame, p, 5, (0, 0, 255), -1, cv2.LINE_AA)

        if self.cfg.draw_text:
            txt = f"Distance: {float(total_distance_m):.1f} m"
            cv2.putText(frame, txt, (12, self._h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        try:
            cv2.imshow(self.cfg.window_name, frame)
            cv2.waitKey(1)
        except Exception:
            self._ready = False

    def close(self):
        if not self._ready:
            return
        try:
            cv2.destroyWindow(self.cfg.window_name)
        except Exception:
            pass
        self._ready = False


def build_gm_background(gm_polylines, bounds, win_size=(520, 520), margin_px=10,
                         bg_color=(10, 10, 10), road_color=(120, 120, 120),
                         flip_x=True, flip_y=False):
    """Pre-render road polylines onto a background image for the global map."""
    w, h = int(win_size[0]), int(win_size[1])
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = bg_color

    if gm_polylines:
        for pts in gm_polylines:
            pix = [
                world_to_gm_px(x, y, bounds, w, h, int(margin_px), flip_x=flip_x, flip_y=flip_y)
                for (x, y) in pts
            ]
            for i in range(len(pix) - 1):
                cv2.line(bg, pix[i], pix[i + 1], road_color, 1, cv2.LINE_AA)

    cv2.rectangle(bg, (0, 0), (w - 1, h - 1), (200, 200, 200), 1)
    return bg
