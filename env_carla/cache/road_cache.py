"""
Road topology cache for pre-sampling map polylines.

Provides polylines at two resolutions: dense for BEV rendering and
coarse for global map rendering. Built once at environment initialisation.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Any

Point2D = Tuple[float, float]
Polyline2D = List[Point2D]
Bounds = Tuple[float, float, float, float]


@dataclass
class RoadCacheConfig:
    bev_step_m: float = 4.0
    gm_step_m: float = 8.0
    gm_bounds_pad_m: float = 20.0
    max_iter_per_segment: int = 8000
    fallback_bounds: Bounds = (-1.0, 1.0, -1.0, 1.0)


class RoadCache:
    """Cache of map topology sampled as polylines for rendering."""

    def __init__(self, carla_map, config=None):
        self._map = carla_map
        self.cfg = config or RoadCacheConfig()
        self._bev_polylines = []
        self._gm_polylines = []
        self._gm_bounds = None
        self._built = False

    @property
    def bev_polylines(self):
        return self._bev_polylines

    @property
    def gm_polylines(self):
        return self._gm_polylines

    @property
    def gm_bounds(self):
        return self._gm_bounds

    def build(self):
        """Sample the map topology into polylines."""
        topo = self._safe_get_topology()

        self._bev_polylines = self._sample_topology_polylines(
            topo, step_m=float(max(0.5, self.cfg.bev_step_m)), with_bounds=False,
        )

        gm_polylines, bounds = self._sample_topology_polylines_with_bounds(
            topo, step_m=float(max(0.5, self.cfg.gm_step_m)),
            pad=float(max(0.0, self.cfg.gm_bounds_pad_m)),
        )
        self._gm_polylines = gm_polylines
        self._gm_bounds = bounds
        self._built = True

    def rebuild(self, config=None):
        if config is not None:
            self.cfg = config
        self.build()

    def _safe_get_topology(self):
        try:
            topo = self._map.get_topology()
            if topo is None:
                return []
            return topo
        except Exception:
            return []

    def _sample_topology_polylines(self, topo, step_m, with_bounds):
        polylines = []
        for a, b in topo:
            try:
                pts = self._sample_segment(a, b, step_m=step_m)
                if len(pts) >= 2:
                    polylines.append(pts)
            except Exception:
                continue
        if with_bounds:
            xs = [p[0] for poly in polylines for p in poly]
            ys = [p[1] for poly in polylines for p in poly]
            if not xs or not ys:
                return polylines, self.cfg.fallback_bounds
            minx, maxx = float(min(xs)), float(max(xs))
            miny, maxy = float(min(ys)), float(max(ys))
            return polylines, (minx, maxx, miny, maxy)
        return polylines

    def _sample_topology_polylines_with_bounds(self, topo, step_m, pad):
        polylines = []
        xs = []
        ys = []
        for a, b in topo:
            try:
                pts = self._sample_segment(a, b, step_m=step_m)
                if len(pts) >= 2:
                    polylines.append(pts)
                    for x, y in pts:
                        xs.append(float(x))
                        ys.append(float(y))
            except Exception:
                continue
        if not xs or not ys:
            return polylines, self.cfg.fallback_bounds
        minx, maxx = float(min(xs)), float(max(xs))
        miny, maxy = float(min(ys)), float(max(ys))
        bounds = (minx - pad, maxx + pad, miny - pad, maxy + pad)
        return polylines, bounds

    def _sample_segment(self, wp_start, wp_end, step_m):
        pts = []
        cur = wp_start
        end_loc = wp_end.transform.location
        safety = 0
        max_iter = int(max(1, self.cfg.max_iter_per_segment))

        while safety < max_iter:
            loc = cur.transform.location
            pts.append((float(loc.x), float(loc.y)))

            try:
                dist = float(loc.distance(end_loc))
            except Exception:
                dx = float(loc.x) - float(end_loc.x)
                dy = float(loc.y) - float(end_loc.y)
                dist = math.hypot(dx, dy)

            if dist < step_m:
                pts.append((float(end_loc.x), float(end_loc.y)))
                break

            nxt = cur.next(step_m)
            if not nxt:
                break
            cur = nxt[0]
            safety += 1

        return pts
