# -*- coding: utf-8 -*-
"""
Utility functions for the CARLA environment.

Provides geometry helpers, waypoint processing, curvature preview computation,
camera projection, lane drawing, semantic compositing, and spawn point lookup.
"""

import cv2
import numpy as np
import math
import carla


def wrap_deg_180(a: float) -> float:
    """Wrap angle in degrees to [-180, 180)."""
    a = float(a)
    return (a + 180.0) % 360.0 - 180.0


def yaw_to_forward_2d(yaw_deg: float):
    """Convert yaw (degrees) to a 2D forward unit vector."""
    r = math.radians(float(yaw_deg))
    return math.cos(r), math.sin(r)


def signed_cross_2d(ax, ay, bx, by) -> float:
    """Compute the signed cross product of two 2D vectors."""
    return float(ax) * float(by) - float(ay) * float(bx)


def safe_call(fn, *args, **kwargs):
    """Call a function, returning None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def normalize_waypoints(wps):
    """Extract bare waypoint objects from a list that may contain (wp, opt) tuples."""
    if wps is None:
        return []
    if len(wps) == 0:
        return []
    if isinstance(wps[0], (tuple, list)):
        wps = [w[0] for w in wps]
    wps = [w for w in wps if hasattr(w, "transform") and w.transform is not None]
    return wps


def compute_speed_norm(ego_velocity):
    """Compute vehicle speed (m/s) from a CARLA velocity vector."""
    if ego_velocity is None:
        return 0.0
    return float(math.hypot(float(ego_velocity.x), float(ego_velocity.y)))


def compute_lat_and_heading_error_from_wp(ego_loc, ego_yaw_deg, ref_wp, sign_convention="right_positive"):
    """Compute lateral offset and heading error relative to a reference waypoint."""
    if ego_loc is None or ref_wp is None:
        return 0.0, 0.0

    try:
        wp_loc = ref_wp.transform.location
        wp_yaw = float(ref_wp.transform.rotation.yaw)
    except Exception:
        return 0.0, 0.0

    fx, fy = yaw_to_forward_2d(wp_yaw)
    ex = float(ego_loc.x) - float(wp_loc.x)
    ey = float(ego_loc.y) - float(wp_loc.y)
    c = signed_cross_2d(fx, fy, ex, ey)

    if str(sign_convention).lower().strip() == "right_positive":
        lat_err = c
    else:
        lat_err = -c

    heading_err = wrap_deg_180(float(ego_yaw_deg) - float(wp_yaw))
    return float(lat_err), float(heading_err)


def _yaw_from_points(x0, y0, x1, y1):
    return math.degrees(math.atan2(float(y1) - float(y0), float(x1) - float(x0)))


def compute_turn_prompt_from_waypoints(wps, horizon_m=5.0, yaw_thresh_deg=15.0, min_pts=4):
    """Detect upcoming turn direction: -1 left, 1 right, 0 straight."""
    wps = normalize_waypoints(wps)
    if wps is None or len(wps) < max(2, int(min_pts)):
        return 0

    horizon_m = float(max(0.5, horizon_m))
    yaw_thresh_deg = float(max(1.0, yaw_thresh_deg))

    pts = []
    for wp in wps[: max(10, int(min_pts) + 2)]:
        try:
            loc = wp.transform.location
            pts.append((float(loc.x), float(loc.y)))
        except Exception:
            continue
    if len(pts) < 2:
        return 0

    start_yaw = _yaw_from_points(pts[0][0], pts[0][1], pts[1][0], pts[1][1])

    acc = 0.0
    end_yaw = start_yaw
    prev = pts[0]
    for i in range(1, len(pts)):
        cur = pts[i]
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            prev = cur
            continue
        acc += seg
        end_yaw = _yaw_from_points(prev[0], prev[1], cur[0], cur[1])
        prev = cur
        if acc >= horizon_m:
            break

    dyaw = wrap_deg_180(end_yaw - start_yaw)
    if dyaw > yaw_thresh_deg:
        return -1
    if dyaw < -yaw_thresh_deg:
        return 1
    return 0


def compute_curvature_preview(wps, ego_x, ego_y, ego_yaw_rad,
                               preview_distances=(1, 3, 5),
                               max_lateral=15.0, max_heading_deg=90.0):
    """Compute 6-dim curvature features at preview distances ahead."""
    N = len(preview_distances)
    result = np.zeros(2 * N, dtype=np.float32)

    wps = normalize_waypoints(wps)
    if wps is None or len(wps) < 2:
        return result

    raw_pts = []
    for wp in wps[:600]:
        try:
            loc = wp.transform.location
            raw_pts.append((float(loc.x), float(loc.y)))
        except Exception:
            continue
    if len(raw_pts) < 2:
        return result

    filtered_pts = [raw_pts[0]]
    cum_dist = [0.0]
    yaws = []
    prev = raw_pts[0]
    for i in range(1, len(raw_pts)):
        cur = raw_pts[i]
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg < 1e-6:
            continue
        cum_dist.append(cum_dist[-1] + seg)
        filtered_pts.append(cur)
        yaws.append(_yaw_from_points(prev[0], prev[1], cur[0], cur[1]))
        prev = cur
        if cum_dist[-1] >= max(preview_distances) + 5:
            break

    if len(filtered_pts) < 2:
        return result

    max_dist = cum_dist[-1]
    ego_yaw_deg = math.degrees(ego_yaw_rad)
    cos_neg_yaw = math.cos(-ego_yaw_rad)
    sin_neg_yaw = math.sin(-ego_yaw_rad)

    laterals = []
    headings = []

    for d in preview_distances:
        if d > max_dist or len(cum_dist) < 2:
            laterals.append(0.0)
            headings.append(0.0)
            continue

        lo, hi = 0, len(cum_dist) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cum_dist[mid] < d:
                lo = mid
            else:
                hi = mid

        d0, d1 = cum_dist[lo], cum_dist[hi]
        t = (d - d0) / max(d1 - d0, 1e-6)
        t = max(0.0, min(1.0, t))

        wx = filtered_pts[lo][0] + t * (filtered_pts[hi][0] - filtered_pts[lo][0])
        wy = filtered_pts[lo][1] + t * (filtered_pts[hi][1] - filtered_pts[lo][1])

        dx_w = wx - ego_x
        dy_w = wy - ego_y
        local_y = dx_w * sin_neg_yaw + dy_w * cos_neg_yaw

        lat = float(np.clip(local_y / max(max_lateral, 1.0), -1.0, 1.0))
        laterals.append(lat)

        yaw_idx = min(hi - 1, len(yaws) - 1) if hi > 0 else 0
        yaw_at_d_deg = yaws[yaw_idx] if yaws else ego_yaw_deg
        dyaw = wrap_deg_180(yaw_at_d_deg - ego_yaw_deg)
        head = float(np.clip(dyaw / max(max_heading_deg, 1.0), -1.0, 1.0))
        headings.append(head)

    result[:N] = laterals
    result[N:] = headings
    return result


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """Build a 3x3 camera intrinsic matrix from image size and FoV."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def _project_world_to_img(Pw, world_2_camera, K):
    Pc = world_2_camera @ Pw
    depth = float(Pc[0])
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if depth <= 1e-6:
        return None, None, depth

    u = fx * (float(Pc[1]) / depth) + cx
    v = fy * (-float(Pc[2]) / depth) + cy
    return u, v, depth


def _resample_xyz(xyz, ds=0.5):
    if xyz is None or len(xyz) < 2:
        return xyz
    pts = np.asarray(xyz, dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total < 1e-6:
        return xyz

    s_new = np.arange(0.0, total, float(ds))
    out = []
    j = 0
    for sn in s_new:
        while j + 1 < len(s) and s[j + 1] < sn:
            j += 1
        denom = max(s[j + 1] - s[j], 1e-9)
        t = (sn - s[j]) / denom
        p = pts[j] + t * (pts[j + 1] - pts[j])
        out.append((float(p[0]), float(p[1]), float(p[2])))

    out.append((float(pts[-1, 0]), float(pts[-1, 1]), float(pts[-1, 2])))
    return out


def draw_lane(image, waypoints, world_2_camera, K, max_dist=60, ego_z=None):
    """Project lane boundaries onto an image using camera projection."""
    H, W = image.shape[:2]
    if waypoints is None or len(waypoints) < 2:
        return image

    LANE_HALF_WIDTH_M = 1.2
    Z_OFFSET = -0.35
    DS = 0.5

    centers = []
    rights = []
    for wp in waypoints:
        loc = wp.transform.location
        rv = wp.transform.get_right_vector()
        centers.append((float(loc.x), float(loc.y), float(loc.z) + float(Z_OFFSET)))
        rights.append((float(rv.x), float(rv.y), float(rv.z)))

    centers = _resample_xyz(centers, ds=DS)
    if centers is None or len(centers) < 2:
        return image

    if len(rights) >= 2:
        rights_rs = []
        n0 = len(rights)
        n1 = len(centers)
        for i in range(n1):
            j = int(round(i * (n0 - 1) / max(n1 - 1, 1)))
            j = max(0, min(n0 - 1, j))
            rx, ry, rz = rights[j]
            Lr = math.sqrt(rx * rx + ry * ry + rz * rz)
            if Lr < 1e-6:
                rx, ry, rz = 0.0, 1.0, 0.0
                Lr = 1.0
            rights_rs.append((rx / Lr, ry / Lr, rz / Lr))
    else:
        rights_rs = [(0.0, 1.0, 0.0)] * len(centers)

    left_uv = []
    right_uv = []

    started = False
    DEPTH_NEAR = 0.15

    for i, (x, y, z) in enumerate(centers):
        rx, ry, rz = rights_rs[i]

        lx = x - rx * LANE_HALF_WIDTH_M
        ly = y - ry * LANE_HALF_WIDTH_M
        lz = z - rz * LANE_HALF_WIDTH_M

        rx2 = x + rx * LANE_HALF_WIDTH_M
        ry2 = y + ry * LANE_HALF_WIDTH_M
        rz2 = z + rz * LANE_HALF_WIDTH_M

        uL, vL, dL = _project_world_to_img(
            np.array([lx, ly, lz, 1.0], dtype=np.float32), world_2_camera, K
        )
        uR, vR, dR = _project_world_to_img(
            np.array([rx2, ry2, rz2, 1.0], dtype=np.float32), world_2_camera, K
        )

        if dL <= DEPTH_NEAR or dR <= DEPTH_NEAR:
            if not started:
                continue
            else:
                break

        if dL > max_dist and dR > max_dist:
            break

        started = True
        left_uv.append((uL, vL))
        right_uv.append((uR, vR))

    if len(left_uv) < 2 or len(right_uv) < 2:
        return image

    poly = np.array(left_uv + right_uv[::-1], dtype=np.int32)

    if poly[:, 0].max() < 0 or poly[:, 0].min() >= W or poly[:, 1].max() < 0 or poly[:, 1].min() >= H:
        return image

    cv2.fillPoly(image, [poly], (0, 255, 255))
    return image


def bbox_to_polygon_from_actor(actor):
    """Extract the 2D bounding box polygon of a CARLA actor."""
    trans = actor.get_transform()
    bb = actor.bounding_box
    x = float(trans.location.x)
    y = float(trans.location.y)
    yaw = float(trans.rotation.yaw) / 180.0 * math.pi
    l = float(bb.extent.x)
    w = float(bb.extent.y)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    p1x, p1y = l, w
    p2x, p2y = l, -w
    p3x, p3y = -l, -w
    p4x, p4y = -l, w

    def rot(px, py):
        rx = cy * px - sy * py + x
        ry = sy * px + cy * py + y
        return [rx, ry]

    return np.array([rot(p1x, p1y), rot(p2x, p2y), rot(p3x, p3y), rot(p4x, p4y)], dtype=np.float32)


def get_spawn_for_task(task_mode):
    """Return fixed spawn location and rotation for predefined tasks."""
    if task_mode == "roundabout":
        return (
            carla.Location(x=-84.5, y=-61.5),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
        )
    if task_mode == "highway":
        return (
            carla.Location(x=-415.5, y=34),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
    return None, None


def parse_location_like(x, default_z):
    """Convert various location formats to carla.Location."""
    if isinstance(x, carla.Location):
        return x
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return carla.Location(
            x=float(x[0]),
            y=float(x[1]),
            z=float(x[2]) if len(x) > 2 else float(default_z),
        )
    return None


def compose_semantic_with_lane(semantic_hwc, semantic_raw_hwc, semantic_lane_hwc,
                                bg_color_dict, vehicle_color_dict):
    """Composite lane overlay onto semantic segmentation, preserving vehicle pixels."""
    semantic = semantic_hwc
    semantic_raw = semantic_raw_hwc
    semantic_lane = semantic_lane_hwc

    for _, value in bg_color_dict.items():
        mask = (semantic == value).all(axis=-1)
        semantic[mask] = [0, 0, 0]

    H, W = semantic.shape[0], semantic.shape[1]
    m = np.zeros((H, W), dtype=bool)
    for _, value in vehicle_color_dict.items():
        m |= (semantic_raw == value).all(axis=-1)

    lane_mask = (semantic_lane[:, :, 0] != 0) | (semantic_lane[:, :, 1] != 0) | (semantic_lane[:, :, 2] != 0)
    overlay_mask = lane_mask & (~m)
    semantic[overlay_mask] = semantic_lane[overlay_mask]

    semantic[m] = semantic_raw[m]
    return semantic


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """Check if a target location is within a given distance ahead of the ego."""
    target_vec = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])

    norm = np.linalg.norm(target_vec)
    if norm > max_distance or norm < 1e-6:
        return False

    forward = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])

    angle = math.degrees(math.acos(np.clip(np.dot(forward, target_vec) / norm, -1.0, 1.0)))
    return angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """Compute distance and angle between two locations relative to an orientation."""
    target_vec = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])

    norm = np.linalg.norm(target_vec)
    if norm < 1e-6:
        return 0.0, 0.0
    forward = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])

    angle = math.degrees(math.acos(np.clip(np.dot(forward, target_vec) / norm, -1.0, 1.0)))
    return norm, angle
