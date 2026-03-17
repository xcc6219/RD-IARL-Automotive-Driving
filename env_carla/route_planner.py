# -*- coding: utf-8 -*-
"""
Route planner for the CARLA environment.

Builds a global route plan using CARLA's GlobalRoutePlanner (A*-based) and
provides per-step waypoint windows, closest-waypoint tracking, hazard
detection (traffic lights, vehicles ahead), and turn prompt generation.
"""

from enum import Enum
from env_carla.utils import *
import math
import carla

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    _HAS_GRP = True
except Exception:
    GlobalRoutePlanner = None
    GlobalRoutePlannerDAO = None
    _HAS_GRP = False


class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


def _map_agents_road_option(opt):
    if opt is None:
        return RoadOption.LANEFOLLOW
    s = str(opt).upper()
    if "LEFT" in s:
        return RoadOption.LEFT
    if "RIGHT" in s:
        return RoadOption.RIGHT
    if "STRAIGHT" in s:
        return RoadOption.STRAIGHT
    if "LANEFOLLOW" in s or "FOLLOW" in s:
        return RoadOption.LANEFOLLOW
    return RoadOption.LANEFOLLOW


def _wrap_deg_180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def compute_connection(current_waypoint, next_waypoint, straight_thresh_deg=10.0, turn_thresh_deg=35.0):
    """Determine the road option connecting two waypoints."""
    c = float(current_waypoint.transform.rotation.yaw)
    n = float(next_waypoint.transform.rotation.yaw)
    d = _wrap_deg_180(n - c)
    ad = abs(d)
    if ad < straight_thresh_deg:
        return RoadOption.STRAIGHT
    if ad < turn_thresh_deg:
        return RoadOption.LANEFOLLOW
    return RoadOption.LEFT if d > 0.0 else RoadOption.RIGHT


def retrieve_options(list_waypoints, current_waypoint):
    """Compute road options for a list of candidate next waypoints."""
    options = []
    for next_waypoint in list_waypoints:
        nxt2 = next_waypoint.next(3.0)
        next_next_waypoint = nxt2[0] if nxt2 else next_waypoint
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)
    return options


def _pick_next_waypoint(cur_wp, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    cyaw = float(cur_wp.transform.rotation.yaw)
    best = None
    best_abs = None
    for w in candidates:
        nyaw = float(w.transform.rotation.yaw)
        d = _wrap_deg_180(nyaw - cyaw)
        ad = abs(d)
        if best_abs is None or ad < best_abs:
            best_abs = ad
            best = w
    return best


class RoutePlanner:
    """
    Global route planner wrapping CARLA's A* search.

    Maintains a global plan as [(waypoint, RoadOption), ...], tracks the
    closest waypoint to the ego vehicle, and provides hazard detection.
    """

    def __init__(self, vehicle, town_map, buffer_size, destination=None,
                 sampling_resolution=0.2, default_forward_distance=300.0,
                 replan_on_lane_change=False, route_step_dist=None):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = town_map
        self._buffer_size = int(buffer_size)
        self._sampling_resolution = float(max(0.1, sampling_resolution))
        self._default_forward_distance = float(default_forward_distance)

        if route_step_dist is None:
            route_step_dist = 1.0
        self._route_step_dist = float(max(0.5, route_step_dist))

        self._closest_search_back = 80
        self._closest_search_forward = 400
        self._last_traffic_light = None
        self._proximity_threshold = 10.0
        self._target_waypoint = None
        self._replan_on_lane_change = bool(replan_on_lane_change)
        self._previous_lane_id = None
        self._cached_closest_wp = None
        self._cached_progress = 0.0
        self._cached_frame = -1
        self._spawn_location = self._vehicle.get_location()
        self._fixed_destination = destination if isinstance(destination, carla.Location) else None
        self._global_plan = []
        self._cursor = 0

        if self._fixed_destination is None:
            self._fixed_destination = self._make_default_destination(self._spawn_location, self._default_forward_distance)

        self._build_global_plan(self._spawn_location)

    def set_destination(self, destination_location):
        if isinstance(destination_location, carla.Location):
            self._fixed_destination = destination_location

    def get_destination(self):
        return self._fixed_destination

    def get_global_plan(self):
        return [p[0] for p in self._global_plan] if self._global_plan else []

    def get_global_plan_with_opts(self):
        return list(self._global_plan)

    def get_progress_ratio(self):
        _, prog = self.get_closest_wp_and_progress()
        return float(prog)

    def get_cursor(self):
        return int(self._cursor)

    def replan_from_current(self):
        start = self._vehicle.get_location()
        self._build_global_plan(start)

    def _route_length_m(self, plan_wp):
        if plan_wp is None or len(plan_wp) < 2:
            return 0.0
        s = 0.0
        for i in range(len(plan_wp) - 1):
            a = plan_wp[i].transform.location
            b = plan_wp[i + 1].transform.location
            s += float(a.distance(b))
        return float(s)

    def _trim_route_to_length(self, plan_wp, plan_opt, target_len_m):
        if plan_wp is None or len(plan_wp) < 2:
            return plan_wp, plan_opt

        target_len_m = float(target_len_m)
        if target_len_m <= 0.0:
            out_wp = [plan_wp[0]]
            out_opt = [plan_opt[0]] if plan_opt else [RoadOption.LANEFOLLOW]
            return out_wp, out_opt

        out_wp = [plan_wp[0]]
        out_opt = [plan_opt[0]] if plan_opt else [RoadOption.LANEFOLLOW]

        acc = 0.0
        for i in range(len(plan_wp) - 1):
            a = plan_wp[i].transform.location
            b = plan_wp[i + 1].transform.location
            seg = float(a.distance(b))
            if seg < 1e-6:
                continue

            out_wp.append(plan_wp[i + 1])
            if plan_opt and i + 1 < len(plan_opt):
                out_opt.append(plan_opt[i + 1])
            else:
                out_opt.append(out_opt[-1])

            acc += seg
            if acc >= target_len_m:
                break

        return out_wp, out_opt

    def _extend_route_to_length(self, plan_wp, plan_opt, target_len_m, step_dist=0.2, max_steps=20000):
        if plan_wp is None or len(plan_wp) < 1:
            return plan_wp, plan_opt

        target_len_m = float(target_len_m)
        if target_len_m <= 0.0:
            return plan_wp, plan_opt

        cur_len = self._route_length_m(plan_wp)
        if cur_len >= target_len_m:
            return self._trim_route_to_length(plan_wp, plan_opt, target_len_m)

        step_dist = float(max(0.1, step_dist))

        cur = plan_wp[-1]
        if cur is None:
            return plan_wp, plan_opt

        visited = set()

        def _key(w):
            try:
                loc = w.transform.location
                return (int(w.road_id), int(w.lane_id), int(loc.x), int(loc.y))
            except Exception:
                return None

        for _ in range(int(max_steps)):
            if cur_len >= target_len_m:
                break

            try:
                nxt = cur.next(step_dist)
            except Exception:
                break
            if not nxt:
                break

            cand = _pick_next_waypoint(cur, nxt)
            if cand is None:
                break

            k = _key(cand)
            if k is not None:
                if k in visited:
                    break
                visited.add(k)

            plan_wp.append(cand)
            if plan_opt:
                plan_opt.append(plan_opt[-1])
            else:
                plan_opt = [RoadOption.LANEFOLLOW] * len(plan_wp)

            try:
                cur_len += float(cur.transform.location.distance(cand.transform.location))
            except Exception:
                cur_len = self._route_length_m(plan_wp)

            cur = cand

        plan_wp, plan_opt = self._trim_route_to_length(plan_wp, plan_opt, target_len_m)
        return plan_wp, plan_opt

    def _build_global_plan(self, start_location):
        self._global_plan = []
        self._cursor = 0

        target_len = float(self._default_forward_distance if self._default_forward_distance is not None else 300.0)
        target_len = max(1.0, target_len)

        tmp_wp = []
        tmp_opt = []

        if _HAS_GRP:
            try:
                dao = GlobalRoutePlannerDAO(self._map, self._sampling_resolution)
                grp = GlobalRoutePlanner(dao)
                grp.setup()

                if self._fixed_destination is None:
                    self._fixed_destination = self._make_default_destination(start_location, target_len)

                route = grp.trace_route(start_location, self._fixed_destination)
                for wp, opt in route:
                    if wp is None:
                        continue
                    tmp_wp.append(wp)
                    tmp_opt.append(_map_agents_road_option(opt))
            except Exception:
                tmp_wp = []
                tmp_opt = []

        if len(tmp_wp) < 2:
            plan_wp, plan_opt = self._fallback_build_by_next(
                start_location, self._fixed_destination, target_len=target_len
            )
            self._global_plan = list(zip(plan_wp, plan_opt))
            self._target_waypoint = self._global_plan[0][0] if self._global_plan else None
            return

        tmp_wp, tmp_opt = self._extend_route_to_length(
            tmp_wp, tmp_opt,
            target_len_m=target_len,
            step_dist=self._sampling_resolution,
        )

        self._global_plan = list(zip(tmp_wp, tmp_opt))

        try:
            self._fixed_destination = self._global_plan[-1][0].transform.location
        except Exception:
            pass

        self._target_waypoint = self._global_plan[0][0] if self._global_plan else None

    def _fallback_build_by_next(self, start_location, destination_location, target_len=None):
        start_wp = self._map.get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            return [], []

        plan_wp = [start_wp]
        plan_opt = [RoadOption.LANEFOLLOW]

        target_len = float(target_len if target_len is not None else self._default_forward_distance)
        target_len = max(1.0, target_len)

        step_dist = float(self._route_step_dist)
        max_steps = 20000

        cur = start_wp
        for _ in range(int(max_steps)):
            if self._route_length_m(plan_wp) >= target_len:
                break

            nxt = cur.next(step_dist)
            if not nxt:
                break

            if len(nxt) == 1:
                cur = nxt[0]
                plan_wp.append(cur)
                plan_opt.append(RoadOption.LANEFOLLOW)
            else:
                road_options_list = retrieve_options(nxt, cur)
                priority = [RoadOption.LANEFOLLOW, RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]
                chosen_idx = 0
                for po in priority:
                    if po in road_options_list:
                        chosen_idx = road_options_list.index(po)
                        break
                cand = nxt[chosen_idx]
                cur = cand
                plan_wp.append(cur)
                plan_opt.append(RoadOption.LANEFOLLOW)

        plan_wp, plan_opt = self._extend_route_to_length(
            plan_wp, plan_opt, target_len_m=target_len, step_dist=step_dist
        )

        try:
            self._fixed_destination = plan_wp[-1].transform.location
        except Exception:
            pass

        return plan_wp, plan_opt

    def _make_default_destination(self, start_location, forward_distance):
        wp = self._map.get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return start_location

        dist = 0.0
        step = float(self._route_step_dist)
        max_iter = int(max(1.0, float(forward_distance) / step)) + 80
        cur = wp
        for _ in range(max_iter):
            nxt = cur.next(step)
            if not nxt:
                break
            cand = _pick_next_waypoint(cur, nxt)
            if cand is None:
                break
            cur = cand
            dist += step
            if dist >= float(forward_distance):
                break
        return cur.transform.location

    def get_closest_wp_and_progress(self):
        """Return the closest waypoint on the global plan and a progress ratio."""
        if self._cached_closest_wp is not None and self._cached_frame >= 0:
            return self._cached_closest_wp, self._cached_progress

        wp, prog = self._compute_closest_wp_and_progress()
        self._cached_closest_wp = wp
        self._cached_progress = prog
        self._cached_frame = 1
        return wp, prog

    def invalidate_cache(self):
        self._cached_closest_wp = None
        self._cached_progress = 0.0
        self._cached_frame = -1

    def _compute_closest_wp_and_progress(self):
        if not self._global_plan:
            wp0 = self._map.get_waypoint(
                self._vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp0 is None:
                return None, 0.0
            self._global_plan = [(wp0, RoadOption.LANEFOLLOW)]
            self._cursor = 0
            return wp0, 0.0

        ego_loc = self._vehicle.get_location()
        n = len(self._global_plan)
        cur = int(max(0, min(self._cursor, n - 1)))

        start = max(0, cur - int(self._closest_search_back))
        end = min(n, cur + int(self._closest_search_forward))

        best_idx = cur
        best_d2 = None

        ex = float(ego_loc.x)
        ey = float(ego_loc.y)

        for i in range(start, end):
            wp = self._global_plan[i][0]
            loc = wp.transform.location
            dx = float(loc.x) - ex
            dy = float(loc.y) - ey
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i

        self._cursor = int(best_idx)
        wp_best = self._global_plan[self._cursor][0]
        prog = float(self._cursor) / float(max(n - 1, 1))
        return wp_best, prog

    def run_step(self):
        """Advance one step: return (waypoints_window, red_light, vehicle_front, vehicle_dist)."""
        self.invalidate_cache()

        if self._replan_on_lane_change:
            try:
                current_wp = self._map.get_waypoint(
                    self._vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if current_wp is not None:
                    lane_id = current_wp.lane_id
                    if self._previous_lane_id is not None and lane_id != self._previous_lane_id:
                        self.replan_from_current()
                    self._previous_lane_id = lane_id
            except Exception:
                pass

        window = self._get_waypoints_window_from_global_plan()
        red_light, vehicle_front, vehicle_dist = self._get_hazard()
        return window, red_light, vehicle_front, vehicle_dist

    def _get_waypoints_window_from_global_plan(self):
        self.get_closest_wp_and_progress()

        if not self._global_plan:
            return []

        end = min(len(self._global_plan), self._cursor + self._buffer_size)
        window = self._global_plan[self._cursor:end]

        if window:
            self._target_waypoint = window[0][0]
        else:
            self._target_waypoint = self._global_plan[-1][0]
            window = [self._global_plan[-1]]

        return window

    def get_future_turn_prompt(self, horizon_m=5.0, yaw_thresh_deg=15.0):
        """Detect upcoming turns from route options: -1 left, 1 right, 0 straight."""
        horizon_m = float(max(0.5, horizon_m))
        yaw_thresh_deg = float(max(1.0, yaw_thresh_deg))

        if not self._global_plan or len(self._global_plan) < 3:
            return 0

        cur = int(max(0, min(self._cursor, len(self._global_plan) - 2)))

        acc = 0.0
        for i in range(cur, len(self._global_plan) - 1):
            a = self._global_plan[i][0].transform.location
            b = self._global_plan[i + 1][0].transform.location
            seg = float(a.distance(b))
            if seg < 1e-6:
                continue
            acc += seg
            opt = self._global_plan[i + 1][1]
            if opt == RoadOption.LEFT:
                return -1
            if opt == RoadOption.RIGHT:
                return 1
            if acc >= horizon_m:
                break
        return 0

    def _get_hazard(self):
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        vehicle_state, vehicle_dist = self._is_vehicle_hazard(vehicle_list)
        light_state = self._is_light_red_us_style(lights_list)
        self._cached_vehicle_list = vehicle_list
        del actor_list, lights_list
        return light_state, vehicle_state, vehicle_dist

    def get_cached_vehicle_list(self):
        """Return vehicle list cached from the last run_step() call."""
        return getattr(self, '_cached_vehicle_list', None)

    def _is_vehicle_hazard(self, vehicle_list):
        """Detect same-lane vehicles ahead within proximity threshold."""
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(
            ego_vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_vehicle_waypoint is None:
            return False, float(self._proximity_threshold)

        closest_dist = float(self._proximity_threshold)
        found = False
        ego_yaw = self._vehicle.get_transform().rotation.yaw

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            try:
                target_vehicle_waypoint = self._map.get_waypoint(
                    target_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if target_vehicle_waypoint is None:
                    continue
                if (
                    target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id
                    or target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
                ):
                    continue
                loc = target_vehicle.get_location()
                dx = float(loc.x - ego_vehicle_location.x)
                dy = float(loc.y - ego_vehicle_location.y)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > self._proximity_threshold or dist < 1e-6:
                    continue
                forward_x = math.cos(math.radians(ego_yaw))
                forward_y = math.sin(math.radians(ego_yaw))
                dot = (dx * forward_x + dy * forward_y) / dist
                if dot > 0.0:
                    if dist < closest_dist:
                        closest_dist = dist
                        found = True
            except Exception:
                continue
        return found, closest_dist

    def _is_light_red_us_style(self, lights_list):
        """Detect red traffic lights using US-style intersection logic."""
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(
            ego_vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_vehicle_waypoint is None:
            return False
        if ego_vehicle_waypoint.is_intersection:
            return False

        if self._target_waypoint is not None and self._target_waypoint.is_intersection:
            min_angle = 180.0
            sel_traffic_light = None

            for traffic_light in lights_list:
                try:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(
                        loc,
                        ego_vehicle_location,
                        self._vehicle.get_transform().rotation.yaw,
                    )
                    if magnitude < 50.0 and angle < min(25.0, min_angle):
                        sel_traffic_light = traffic_light
                        min_angle = angle
                except Exception:
                    continue

            if sel_traffic_light is not None:
                if self._last_traffic_light is None:
                    self._last_traffic_light = sel_traffic_light
                if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                    return True
            else:
                self._last_traffic_light = None

        return False
