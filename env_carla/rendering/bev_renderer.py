"""
Bird's-eye view renderer for visualising the ego vehicle, road network,
planned route, and surrounding traffic from a top-down perspective.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import cv2

Point2D = Tuple[float, float]
Polyline2D = List[Point2D]
VehiclePolygons = Dict[int, np.ndarray]


@dataclass(frozen=True)
class BevConfig:
    w: int
    h: int
    pixels_per_meter: float = 3.0
    ego_offset_y_ratio: float = 0.7
    road_radius_m: float = 80.0


def world_to_bev_px(wx, wy, ego_x, ego_y, cos_yaw, sin_yaw, cx, cy, ppm):
    """Transform a world coordinate to BEV pixel coordinates."""
    dx = wx - ego_x
    dy = wy - ego_y
    ex = cos_yaw * dx + sin_yaw * dy
    ey = -sin_yaw * dx + cos_yaw * dy
    u = int(round(cx + ey * ppm))
    v = int(round(cy - ex * ppm))
    return u, v, ex, ey


class BevRenderer:
    """Renders a bird's-eye view image centred on the ego vehicle."""

    def __init__(self, config, road_polylines_world=None):
        self.cfg = config
        ego_offset_y_ratio = float(np.clip(self.cfg.ego_offset_y_ratio, 0.55, 0.90))
        self.cx = int(self.cfg.w * 0.5)
        self.cy = int(self.cfg.h * ego_offset_y_ratio)
        self.ppm = float(self.cfg.pixels_per_meter)
        self.road_polylines_world = list(road_polylines_world) if road_polylines_world else []

    def set_road_polylines(self, road_polylines_world):
        self.road_polylines_world = list(road_polylines_world)

    def render(self, ego_transform, waypoints=None, vehicle_polygons=None, ego_id=None):
        """Render BEV image and return as HxWx3 uint8 array."""
        bev = np.zeros((self.cfg.h, self.cfg.w, 3), dtype=np.uint8)

        if ego_transform is None:
            return bev

        ego_loc = ego_transform.location
        ego_x, ego_y = float(ego_loc.x), float(ego_loc.y)

        yaw = math.radians(float(ego_transform.rotation.yaw))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        R = float(self.cfg.road_radius_m)
        R2 = R * R

        if self.road_polylines_world:
            for poly in self.road_polylines_world:
                prev = None
                prev_in = False
                for (wx, wy) in poly:
                    u, v, ex, ey = world_to_bev_px(
                        float(wx), float(wy), ego_x, ego_y,
                        cos_yaw, sin_yaw, self.cx, self.cy, self.ppm,
                    )
                    inside = (ex * ex + ey * ey) <= R2
                    if prev is not None and (inside or prev_in):
                        cv2.line(bev, prev, (u, v), (90, 90, 90), 1, cv2.LINE_AA)
                    prev = (u, v)
                    prev_in = inside

        if waypoints is not None:
            try:
                if len(waypoints) >= 2:
                    pts = []
                    for wp in waypoints:
                        loc = wp.transform.location
                        u, v, _, _ = world_to_bev_px(
                            float(loc.x), float(loc.y), ego_x, ego_y,
                            cos_yaw, sin_yaw, self.cx, self.cy, self.ppm,
                        )
                        pts.append((u, v))
                    for i in range(len(pts) - 1):
                        cv2.line(bev, pts[i], pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        if vehicle_polygons:
            for vid, poly in vehicle_polygons.items():
                try:
                    if ego_id is not None and int(vid) == int(ego_id):
                        continue
                    poly_px = []
                    for corner in poly:
                        wx, wy = float(corner[0]), float(corner[1])
                        u, v, _, _ = world_to_bev_px(
                            wx, wy, ego_x, ego_y,
                            cos_yaw, sin_yaw, self.cx, self.cy, self.ppm,
                        )
                        poly_px.append([u, v])
                    poly_px = np.array([poly_px], dtype=np.int32)
                    cv2.fillPoly(bev, poly_px, (0, 0, 255))
                except Exception:
                    continue

        ego_l_m = 4.5
        ego_w_m = 2.0
        hl = int(0.5 * ego_l_m * self.ppm)
        hw = int(0.5 * ego_w_m * self.ppm)

        poly_px = np.array(
            [[[self.cx - hw, self.cy - hl],
              [self.cx + hw, self.cy - hl],
              [self.cx + hw, self.cy + hl],
              [self.cx - hw, self.cy + hl]]],
            dtype=np.int32)

        cv2.fillPoly(bev, poly_px, (255, 0, 0))
        return bev

    @staticmethod
    def to_chw_float01(bev_hwc_uint8):
        return np.transpose(bev_hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0
