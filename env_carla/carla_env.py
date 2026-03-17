# -*- coding: utf-8 -*-
"""
CARLA gym-like environment for DDPG continuous control.

Wraps the CARLA simulator to provide a step-based interface with
multi-modal observations (ego state, bird's-eye view, semantic segmentation)
and continuous throttle/steer actions.
"""

import logging
import gc
import time
from queue import Empty
import random
import math

import numpy as np
import torch
import pygame
import cv2
cv2.setNumThreads(2)

import carla

from env_carla.sensors import SensorManager
from env_carla.route_planner import RoutePlanner
from env_carla.cache.road_cache import RoadCache, RoadCacheConfig
from env_carla.rendering.bev_renderer import BevRenderer, BevConfig
from env_carla.rendering.global_map_renderer import (
    GlobalMapRenderer,
    GlobalMapConfig,
    build_gm_background,
)
from env_carla.utils import *


def _blue(s: str) -> str:
    return f"\033[94m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


class CarlaEnv:
    """
    CARLA environment for continuous control evaluation.

    Observation dict:
        'state'    : np.float32 (14,)
        'birdseye' : np.float32 (3, bev_h, bev_w)   [0,1]
        'semantic'  : np.float32 (3, img_h, img_w)   [0,1]

    State vector (14-dim):
        0   prev_throttle        [0, 1]
        1   prev_steer           [-1, 1]
        2   prev_brake           [0, 1]
        3   angle_norm           heading deviation normalised [-1, 1]
        4   dev_norm             lateral deviation normalised [-1, 1]
        5   speed_norm           speed normalised [0, 2]
        6   a_long_norm          longitudinal accel normalised [-1, 1]
        7   light                red-light signal {0, 1}
        8-10 curvature_lateral   3 preview distances lateral offsets [-1, 1]
        11-13 curvature_heading  3 preview distances heading changes [-1, 1]

    Action (continuous):
        action[0]  acc  in [-1, 1]  (positive=throttle, negative=brake)
        action[1]  steer in [-1, 1]
    """

    def __init__(self, opt):
        self.opt = opt

        self.turn_horizon_m = getattr(self.opt, "turn_horizon_m", 5.0)
        self.turn_yaw_thresh_deg = getattr(self.opt, "turn_yaw_thresh_deg", 15.0)
        self.lat_sign_convention = getattr(self.opt, "lat_sign_convention", "right_positive")

        self.no_rendering_mode = getattr(self.opt, "no_rendering_mode", False)
        self.render_enabled = not self.no_rendering_mode
        self.enable_spectator = getattr(self.opt, "enable_spectator", True)

        self.max_speed = getattr(self.opt, "max_speed", 10.0)
        self.max_angle = getattr(self.opt, "max_angle", 30.0)
        self.max_lane = getattr(self.opt, "max_lane", 1.0)
        self.out_lane = getattr(self.opt, "out_lane", 5.0)
        self.max_time_episode = getattr(self.opt, "max_time_episode", 1000)
        self.max_past_step = getattr(self.opt, "max_past_step", 3)

        self.acc_range = getattr(self.opt, "acc_range", 1.0)
        self.steer_range = getattr(self.opt, "steer_range", 1.0)

        self.img_size = getattr(self.opt, "img_size")
        self.bev_size = getattr(self.opt, "bev_size")

        self.carla_port = getattr(self.opt, "carla_port")
        self.map_name = getattr(self.opt, "map")
        self.fixed_delta_seconds = getattr(self.opt, "fixed_delta_seconds")
        self.synchronous_mode = getattr(self.opt, "synchronous_mode")

        self.global_map_enable = getattr(self.opt, "global_map_enable", True)
        self.global_map_window_name = getattr(self.opt, "global_map_window_name", "CARLA Global Map")
        self.global_map_window_size = getattr(self.opt, "global_map_window_size", (520, 520))
        self.global_map_margin_px = getattr(self.opt, "global_map_margin_px", 10)
        self.global_map_step_m = getattr(self.opt, "global_map_step_m", 8.0)
        self.global_map_trace_len = getattr(self.opt, "global_map_trace_len", 50000)
        self.global_map_trace_draw = getattr(self.opt, "global_map_trace_draw", 5000)

        self.pixels_per_meter = getattr(self.opt, "pixels_per_meter", 3.0)
        self.bev_ego_offset_y_ratio = getattr(self.opt, "bev_ego_offset_y_ratio", 0.7)
        self.bev_road_radius_m = getattr(self.opt, "bev_road_radius_m", 80.0)
        self.bev_road_step_m = getattr(self.opt, "bev_road_step_m", 4.0)

        self.route_sampling_resolution = getattr(self.opt, "route_sampling_resolution", 0.1)
        self.route_default_forward_distance = getattr(self.opt, "route_default_forward_distance", 300.0)
        self.allow_replan = getattr(self.opt, "allow_replan", False)
        self.replan_on_lane_change = getattr(self.opt, "replan_on_lane_change", False)
        self.route_step_dist = getattr(self.opt, "route_step_dist", 1.0)

        self.semantic_drain_time_budget_s = getattr(self.opt, "semantic_drain_time_budget_s", 0.0025)
        self.semantic_drain_time_budget_max_s = getattr(self.opt, "semantic_drain_time_budget_max_s", 0.015)
        self.semantic_drain_soft_qsize = getattr(self.opt, "semantic_drain_soft_qsize", 6)

        self.poly_update_interval = getattr(self.opt, "poly_update_interval", 2)
        self.gm_update_interval = getattr(self.opt, "gm_update_interval", 3)
        self.render_update_interval = getattr(self.opt, "render_update_interval", 1)

        self.goal_progress_ratio = getattr(self.opt, "goal_progress_ratio", 0.8)
        self.goal_dist_thresh = getattr(self.opt, "goal_dist_thresh", 10.0)

        self.number_of_vehicles = getattr(self.opt, "number_of_vehicles", 0)

        self.mode = getattr(self.opt, "mode", "test")
        self.task_mode = getattr(self.opt, "task_mode", "default")

        self.fixed_destination = getattr(self.opt, "fixed_destination", None)

        self.curvature_preview_dists = getattr(self.opt, "preview_distances", (1, 3, 5))
        self.curvature_max_lateral = getattr(self.opt, "curvature_max_lateral", 15.0)
        self.curvature_max_heading_deg = getattr(self.opt, "curvature_max_heading_deg", 90.0)

        self.max_accel = getattr(self.opt, "max_accel", 10.0)

        self.time_step = 0
        self.total_step = 0
        self.reset_step = 0

        self.lane_change = None
        self._prev_lane_id = None
        self.route_planner = None
        self.waypoints = None
        self.light_front = False
        self.vehicle_front = False
        self.vehicle_front_dist = 10.0

        self.is_turning = False
        self.overtake_viable = True

        self.previous_location = None
        self.total_distance = 0.0
        self.forward_speed = 0.0
        self.lane_change_num = 0

        self.vehicle_polygons = []
        self.collision = False
        self.collision_sensor = None

        self.semantic_sensor = None
        self.semantic_queue = None

        self.destinations = None
        self.current_wpt = None

        self.ego_car = None
        self.ego_transform = None
        self.ego_velocity = None
        self.ego_acc = None
        self.ego_control = None
        self.ego_waypoint = None

        W, H = self.img_size
        self.semantic_data = np.zeros((H, W, 3), dtype=np.uint8)

        self._prev_throttle = 0.0
        self._prev_steer = 0.0
        self._prev_brake = 0.0

        self._stuck_count = 0

        self.client = carla.Client(host="127.0.0.1", port=self.carla_port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        try:
            if self.world.get_map().name != self.map_name:
                self.world = self.client.load_world(self.map_name)
        except Exception:
            self.world = self.client.load_world(self.map_name)

        self.map = self.world.get_map()
        print("Server world connection completed!")

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.settings.no_rendering_mode = self.no_rendering_mode

        self.sensors = SensorManager(self.world)

        self.start, self.rotation = get_spawn_for_task(self.task_mode)

        self._set_mode(self.synchronous_mode)

        self.K = build_projection_matrix(self.img_size[0], self.img_size[1], 60.0)

        seed = getattr(self.opt, "seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                torch.manual_seed(seed)
            except Exception:
                pass

        self._bev_w, self._bev_h = int(self.bev_size[0]), int(self.bev_size[1])

        self._road_cache = RoadCache(
            self.map,
            RoadCacheConfig(
                bev_step_m=float(self.bev_road_step_m),
                gm_step_m=float(self.global_map_step_m),
            ),
        )
        self._road_cache.build()

        self._bev_renderer = BevRenderer(
            BevConfig(
                w=self._bev_w,
                h=self._bev_h,
                pixels_per_meter=float(self.pixels_per_meter),
                ego_offset_y_ratio=float(self.bev_ego_offset_y_ratio),
                road_radius_m=float(self.bev_road_radius_m),
            ),
            road_polylines_world=self._road_cache.bev_polylines,
        )

        self._gm_renderer = None
        if self.render_enabled and self.global_map_enable and (self._road_cache.gm_bounds is not None):
            gm_bg = build_gm_background(
                gm_polylines=self._road_cache.gm_polylines,
                bounds=self._road_cache.gm_bounds,
                win_size=self.global_map_window_size,
                margin_px=int(self.global_map_margin_px),
            )
            self._gm_renderer = GlobalMapRenderer(
                GlobalMapConfig(
                    window_name=str(self.global_map_window_name),
                    win_size=self.global_map_window_size,
                    margin_px=int(self.global_map_margin_px),
                    trace_len=int(self.global_map_trace_len),
                    trace_draw=int(self.global_map_trace_draw),
                ),
                gm_bg=gm_bg,
                bounds=self._road_cache.gm_bounds,
            )

        self._other_vehicle_ids = []
        self._tracked_vehicle_ids = []
        self._route_plan_xy = []
        self._fixed_destination = None
        self._last_valid_wps = []

        if self.render_enabled:
            self._init_pygame_window()

    def _spawn_ego_vehicle(self, vehicle_spawn_points, max_tries=1000):
        if self.task_mode in ("roundabout", "highway") and self.start is not None:
            return self._create_ego_car(carla.Transform(self.start, self.rotation))

        for _ in range(int(max_tries)):
            transform = random.choice(vehicle_spawn_points)
            if self._create_ego_car(transform):
                return True
        return False

    def _should_update(self, interval):
        if interval is None:
            return True
        try:
            interval = int(interval)
        except Exception:
            return True
        if interval <= 1:
            return True
        return (self.total_step % interval) == 0

    def _push_history(self, history_list, item):
        history_list.append(item)
        while len(history_list) > int(self.max_past_step):
            history_list.pop(0)

    def _tick_and_drain_semantic(self):
        self._tick()
        self._drain_semantic_by_frame()

    def _drain_semantic_by_frame(self):
        if self.semantic_queue is None:
            return

        try:
            snap = self.world.get_snapshot()
            target_frame = getattr(snap, "frame", None)
        except Exception:
            target_frame = None

        latest_item = None
        matched_item = None

        try:
            try:
                qsz = int(self.semantic_queue.qsize())
            except Exception:
                qsz = 0

            budget = float(self.semantic_drain_time_budget_s)
            if qsz >= int(self.semantic_drain_soft_qsize):
                budget = min(
                    float(self.semantic_drain_time_budget_max_s),
                    max(budget, float(self.semantic_drain_time_budget_s) * 2.0),
                )

            max_pop = max(8, qsz + 2)

            t0 = time.perf_counter()
            pops = 0
            while pops < max_pop:
                if (time.perf_counter() - t0) >= budget:
                    break
                item = self.semantic_queue.get_nowait()
                pops += 1
                latest_item = item

                if target_frame is not None:
                    frame_id = getattr(item, "frame", None)
                    if frame_id is None:
                        frame_id = getattr(item, "frame_number", None)
                    if frame_id == target_frame:
                        matched_item = item
                        break

        except Empty:
            pass
        except Exception:
            pass

        use_item = matched_item if matched_item is not None else latest_item
        if use_item is not None:
            self._get_semantic_image(use_item)

    def _update_route_info(self):
        if self.route_planner is None:
            self.waypoints = None
            self.light_front = False
            self.vehicle_front = False
            self.vehicle_front_dist = 10.0
            self.destinations = None
            self.is_turning = False
            return

        out = self.route_planner.run_step()
        if out is None:
            self.waypoints = None
            self.light_front = False
            self.vehicle_front = False
            self.vehicle_front_dist = 10.0
            self.destinations = None
            self.is_turning = False
            return

        if isinstance(out, tuple) and len(out) >= 4:
            self.waypoints, self.light_front, self.vehicle_front, self.vehicle_front_dist = out[:4]
        elif isinstance(out, tuple) and len(out) >= 3:
            self.waypoints, self.light_front, self.vehicle_front = out[:3]
            self.vehicle_front_dist = 10.0
        else:
            self.waypoints = None
            self.light_front = False
            self.vehicle_front = False
            self.vehicle_front_dist = 10.0

        self.destinations = self.route_planner.get_destination()

        wps = normalize_waypoints(self.waypoints)
        if len(wps) >= 2:
            self._last_valid_wps = wps

        try:
            turn_prompt = int(
                compute_turn_prompt_from_waypoints(
                    wps=self.waypoints,
                    horizon_m=float(self.turn_horizon_m),
                    yaw_thresh_deg=float(self.turn_yaw_thresh_deg),
                    min_pts=4,
                )
            )
        except Exception:
            turn_prompt = 0

        self.is_turning = (turn_prompt != 0)

    def _check_overtake_viable(self):
        if self.total_step % 5 != 0:
            return

        self.overtake_viable = False

        if not self.vehicle_front or self.vehicle_front_dist > 15.0:
            return
        if bool(self.light_front):
            return
        if self.is_turning:
            return

        ego_wp = self.ego_waypoint
        if ego_wp is None:
            return

        vehicle_list = None
        if self.route_planner is not None:
            vehicle_list = self.route_planner.get_cached_vehicle_list()

        for get_lane in (ego_wp.get_left_lane, ego_wp.get_right_lane):
            try:
                adj_wp = get_lane()
            except Exception:
                continue
            if adj_wp is None:
                continue
            if adj_wp.lane_type != carla.LaneType.Driving:
                continue
            if ego_wp.lane_id * adj_wp.lane_id < 0:
                continue

            if self._is_adjacent_lane_clear(adj_wp, vehicle_list):
                self.overtake_viable = True
                return

    def _is_adjacent_lane_clear(self, adj_wp, vehicle_list=None, check_dist=20.0):
        try:
            ego_loc = self.ego_transform.location
            ego_yaw = self.ego_transform.rotation.yaw
            fwd_x = math.cos(math.radians(ego_yaw))
            fwd_y = math.sin(math.radians(ego_yaw))

            if vehicle_list is None:
                vehicle_list = self.world.get_actors().filter("*vehicle*")

            for v in vehicle_list:
                if v.id == self.ego_car.id:
                    continue
                v_loc = v.get_location()
                dx = float(v_loc.x - ego_loc.x)
                dy = float(v_loc.y - ego_loc.y)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > check_dist:
                    continue
                try:
                    v_wp = self.map.get_waypoint(
                        v_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if v_wp is None:
                        continue
                    if v_wp.road_id == adj_wp.road_id and v_wp.lane_id == adj_wp.lane_id:
                        dot = dx * fwd_x + dy * fwd_y
                        if dot > -5.0:
                            return False
                except Exception:
                    continue
            return True
        except Exception:
            return False

    def _update_vehicle_polygons_history(self, force=False):
        if force or self._should_update(self.poly_update_interval):
            polys = self._get_actor_polygons_tracked()
            self._push_history(self.vehicle_polygons, polys)
        else:
            if len(self.vehicle_polygons) > 0:
                self._push_history(self.vehicle_polygons, self.vehicle_polygons[-1])

    def _maybe_update_spectator(self):
        if not self.enable_spectator:
            return
        spectator = safe_call(self.world.get_spectator)
        if spectator is None:
            return
        tf = safe_call(self.ego_car.get_transform)
        if tf is None:
            return
        loc = tf.location
        safe_call(
            spectator.set_transform,
            carla.Transform((loc + carla.Location(z=30)), carla.Rotation(pitch=-90)),
        )

    def _render_frames(self, birdseye_hwc, semantic_hwc):
        if not self.render_enabled:
            return
        if not self._should_update(self.render_update_interval):
            return

        self._blit_array_to_pygame(birdseye_hwc, (0, 0))
        self._blit_array_to_pygame(semantic_hwc, (self.bev_size[0], 0))

        if self._gm_renderer is not None and self._should_update(self.gm_update_interval):
            if self.ego_transform is not None:
                loc = self.ego_transform.location
                self._gm_renderer.update(
                    (float(loc.x), float(loc.y)), total_distance_m=self.total_distance
                )

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            pygame.display.flip()
        except Exception:
            pass

    def _reset_episode_state(self):
        W, H = self.img_size
        self.semantic_data = np.zeros((H, W, 3), dtype=np.uint8)

        self.semantic_sensor = None
        self.semantic_queue = None
        self.collision_sensor = None
        self.collision = False
        self.current_wpt = None

        self.time_step = 0
        self.reset_step += 1

        self.vehicle_polygons = []
        self._other_vehicle_ids = []
        self._tracked_vehicle_ids = []

        self.ego_car = None
        self.ego_transform = None
        self.ego_velocity = None
        self.ego_acc = None
        self.ego_control = None
        self.ego_waypoint = None

        self.lane_change = None
        self._prev_lane_id = None
        self.total_distance = 0.0
        self.previous_location = None
        self.lane_change_num = 0

        self._prev_throttle = 0.0
        self._prev_steer = 0.0
        self._prev_brake = 0.0

        self._last_valid_wps = []

        self.light_front = False
        self.vehicle_front = False
        self.vehicle_front_dist = 10.0
        self.is_turning = False
        self.overtake_viable = False

        self._stuck_count = 0

    def _init_pygame_window(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.bev_size[0] + self.img_size[0], max(self.bev_size[1], self.img_size[1])),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )

    def _blit_array_to_pygame(self, arr_rgb_hwc, top_left):
        try:
            surf = pygame.surfarray.make_surface(np.transpose(arr_rgb_hwc, (1, 0, 2)))
            self.display.blit(surf, top_left)
        except Exception:
            pass

    def _init_route_plan_xy(self):
        self._route_plan_xy = []
        if self.route_planner is None:
            return
        try:
            plan = self.route_planner.get_global_plan()
        except Exception:
            plan = None
        if not plan:
            return
        out = []
        last = None
        for wp in plan:
            try:
                loc = wp.transform.location
                p = (float(loc.x), float(loc.y))
            except Exception:
                continue
            if last is None:
                out.append(p)
                last = p
            else:
                if (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= 0.25:
                    out.append(p)
                    last = p
        self._route_plan_xy = out

    def _route_progress_ratio(self):
        rp = self.route_planner
        if rp is None:
            return 0.0
        try:
            return float(rp.get_progress_ratio())
        except Exception:
            return 0.0

    def _reached_goal(self):
        if self.route_planner is None:
            return False
        prog = self._route_progress_ratio()
        if prog < float(self.goal_progress_ratio):
            return False
        if self.destinations is None:
            return False
        return self._check_waypoint(self.destinations, thresh=float(self.goal_dist_thresh))

    def reset(self):
        """Destroy actors, respawn ego and traffic, return initial observation."""
        self._destroy_all_actors()
        self._reset_episode_state()

        vehicle_spawn_points = list(self.map.get_spawn_points())

        ok = self._spawn_ego_vehicle(vehicle_spawn_points, max_tries=1000)
        if (not ok) or (self.ego_car is None):
            raise RuntimeError("Failed to spawn ego vehicle.")

        self._create_sensors()
        self._other_vehicle_ids = self._create_others_vehicles(vehicle_spawn_points)
        self._tracked_vehicle_ids = [self.ego_car.id] + list(self._other_vehicle_ids)

        self._tick_and_drain_semantic()

        self.previous_location = self.ego_car.get_location()

        loc0 = self.ego_car.get_transform().location
        self._fixed_destination = parse_location_like(self.fixed_destination, default_z=float(loc0.z))

        self.route_planner = RoutePlanner(
            self.ego_car,
            self.map,
            35,
            destination=self._fixed_destination,
            sampling_resolution=float(self.route_sampling_resolution),
            default_forward_distance=float(self.route_default_forward_distance),
            replan_on_lane_change=(bool(self.allow_replan) and bool(self.replan_on_lane_change)),
            route_step_dist=float(self.route_step_dist),
        )

        self._update_route_info()

        self._stuck_count = 0

        self._init_route_plan_xy()
        if self._gm_renderer is not None:
            self._gm_renderer.set_route_plan(self._route_plan_xy)
            self._gm_renderer.reset_trace()

        self._update_vehicle_polygons_history(force=True)

        return self._get_obs()

    def step(self, action):
        """Apply action, tick world, return (observation, reward, done)."""
        if self.mode == "manual":
            throttle, steer, brake = action.throttle, action.steer, action.brake
        else:
            throttle, steer, brake = self._transform_control(action)

        self.ego_car.apply_control(
            carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        )

        self._tick_and_drain_semantic()
        self._maybe_update_spectator()

        self._update_vehicle_polygons_history(force=False)

        self.time_step += 1
        self.total_step += 1

        self._update_route_info()
        self._check_overtake_viable()

        obs = self._get_obs()

        self._prev_throttle = float(np.clip(throttle, 0.0, 1.0))
        self._prev_steer = float(np.clip(steer, -1.0, 1.0))
        self._prev_brake = float(np.clip(brake, 0.0, 1.0))

        reward = self._get_reward()
        done = self._terminal()
        return obs, reward, done

    def _get_obs(self):
        """Build multi-modal observation dict."""
        self.ego_transform = self.ego_car.get_transform()
        self.ego_velocity = self.ego_car.get_velocity()
        self.ego_acc = self.ego_car.get_acceleration()
        self.ego_control = self.ego_car.get_control()

        self.ego_waypoint = self.map.get_waypoint(
            self.ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        if self.ego_waypoint is not None:
            cur_lane_id = (self.ego_waypoint.road_id, self.ego_waypoint.lane_id)
            if self._prev_lane_id is not None and cur_lane_id != self._prev_lane_id:
                self.lane_change_num += 1
            self._prev_lane_id = cur_lane_id

        current_location = self.ego_transform.location
        if self.previous_location is None:
            self.previous_location = current_location
        self.total_distance += float(current_location.distance(self.previous_location))
        self.previous_location = current_location

        ego_state = self._get_ego_state()

        W, H = self.img_size
        semantic = cv2.resize(self.semantic_data, (W, H), interpolation=cv2.INTER_NEAREST)
        semantic_raw = semantic.copy()

        wps_to_draw = normalize_waypoints(self.waypoints)
        if len(wps_to_draw) < 2:
            wps_to_draw = self._last_valid_wps

        semantic_lane = np.zeros_like(semantic)
        if self.semantic_sensor is not None and len(wps_to_draw) >= 2:
            world_2_camera = np.array(self.semantic_sensor.get_transform().get_inverse_matrix())
            semantic_lane = draw_lane(semantic_lane, wps_to_draw, world_2_camera, self.K, 60)

        bg_color = {
            "sky": [70, 130, 180],
            "building": [70, 70, 70],
            "terrain": [152, 251, 152],
            "vegetation": [107, 142, 35],
            "trafficLight": [250, 170, 30],
            "wall": [102, 102, 156],
            "mountain": [145, 170, 100],
            "pole": [153, 153, 153],
            "static": [110, 190, 160],
            "bridge": [150, 100, 100],
            "water": [45, 60, 150],
            "other": [55, 90, 80],
            "fence": [100, 40, 40],
        }

        vehicle_color = {
            "car": [0, 0, 142],
            "bus": [0, 60, 100],
            "truck": [0, 0, 70],
            "train": [0, 60, 100],
            "motorcycle": [0, 0, 230],
            "bicycle": [119, 11, 32],
        }

        semantic = compose_semantic_with_lane(
            semantic_hwc=semantic,
            semantic_raw_hwc=semantic_raw,
            semantic_lane_hwc=semantic_lane,
            bg_color_dict=bg_color,
            vehicle_color_dict=vehicle_color,
        )

        latest_polys = (
            self.vehicle_polygons[-1]
            if (self.vehicle_polygons and len(self.vehicle_polygons) > 0)
            else None
        )

        birdseye_hwc = self._bev_renderer.render(
            self.ego_transform,
            waypoints=wps_to_draw,
            vehicle_polygons=latest_polys,
            ego_id=self.ego_car.id if self.ego_car is not None else None,
        )

        self._render_frames(birdseye_hwc, semantic)

        birdseye = np.transpose(birdseye_hwc, (2, 0, 1)).astype(np.float32) / 255.0
        semantic_chw = np.transpose(semantic, (2, 0, 1)).astype(np.float32) / 255.0

        return {"state": ego_state, "birdseye": birdseye, "semantic": semantic_chw}

    def _get_ego_state(self):
        """Compute the 14-dim normalised ego state vector."""
        ego_loc = None
        try:
            if self.ego_transform is not None:
                ego_loc = self.ego_transform.location
        except Exception:
            ego_loc = None

        ego_yaw_deg = 0.0
        try:
            if self.ego_transform is not None:
                ego_yaw_deg = float(self.ego_transform.rotation.yaw)
        except Exception:
            ego_yaw_deg = 0.0

        ref_wp = None
        if self.route_planner is not None:
            try:
                ref_wp, _ = self.route_planner.get_closest_wp_and_progress()
            except Exception:
                ref_wp = None

        if ref_wp is not None and ego_loc is not None:
            lat_err, heading_deg = compute_lat_and_heading_error_from_wp(
                ego_loc=ego_loc,
                ego_yaw_deg=ego_yaw_deg,
                ref_wp=ref_wp,
                sign_convention=self.lat_sign_convention,
            )
        else:
            lat_err = 0.0
            heading_deg = 0.0

        self.angle = float(heading_deg)
        self.ego_deviation = float(lat_err)
        self.forward_speed = float(compute_speed_norm(self.ego_velocity))

        yaw = math.radians(float(ego_yaw_deg))
        hx, hy = math.cos(yaw), math.sin(yaw)
        ax = float(self.ego_acc.x) if self.ego_acc is not None else 0.0
        ay = float(self.ego_acc.y) if self.ego_acc is not None else 0.0
        a_long = ax * hx + ay * hy

        light = 1.0 if bool(self.light_front) else 0.0

        max_v = max(float(self.max_speed), 1e-6)
        safe_max_angle = max(float(self.max_angle), 1e-6)

        ego_x = float(ego_loc.x) if ego_loc is not None else 0.0
        ego_y = float(ego_loc.y) if ego_loc is not None else 0.0
        ego_yaw_rad = yaw

        wps_for_curv = normalize_waypoints(self.waypoints)
        if len(wps_for_curv) < 2:
            wps_for_curv = self._last_valid_wps

        curvature_6 = compute_curvature_preview(
            wps_for_curv,
            ego_x, ego_y, ego_yaw_rad,
            preview_distances=self.curvature_preview_dists,
            max_lateral=float(self.curvature_max_lateral),
            max_heading_deg=float(self.curvature_max_heading_deg),
        )

        angle_norm = float(np.clip(self.angle / safe_max_angle, -1.0, 1.0))
        dev_norm = float(np.clip(self.ego_deviation / max(float(self.out_lane), 1.0), -1.0, 1.0))
        speed_norm = float(np.clip(self.forward_speed / max_v, 0.0, 2.0))
        a_long_norm = float(np.clip(a_long / max(float(self.max_accel), 1.0), -1.0, 1.0))

        obs = np.array(
            [
                float(self._prev_throttle),
                float(self._prev_steer),
                float(self._prev_brake),
                angle_norm,
                dev_norm,
                speed_norm,
                a_long_norm,
                float(light),
                float(curvature_6[0]),
                float(curvature_6[1]),
                float(curvature_6[2]),
                float(curvature_6[3]),
                float(curvature_6[4]),
                float(curvature_6[5]),
            ],
            dtype=np.float32,
        )
        return np.round(obs, 4).astype(np.float32)

    def _get_reward(self):
        """Reward computation omitted (not required for evaluation)."""
        return 0.0

    def _terminal(self):
        """Check episode termination conditions."""
        if self.collision:
            return True
        if round(abs(self.ego_deviation), 2) >= float(self.out_lane):
            return True

        angle_limit = float(self.max_angle)
        if self.is_turning:
            angle_limit = max(angle_limit, 120.0)

        if round(abs(self.angle), 2) > angle_limit:
            return True
        if self.time_step >= int(self.max_time_episode):
            return True
        if self.destinations is not None:
            if self._check_waypoint(self.destinations):
                return True
        return False

    def _reward_event(self, fail=-50.0):
        """Reward event details omitted (not required for evaluation)."""
        return 0.0

    def _transform_control(self, action):
        """Map [-1,1] continuous action to throttle / steer / brake."""
        a_acc = float(np.clip(action[0], -1.0, 1.0))
        a_ste = float(np.clip(action[1], -1.0, 1.0))
        throttle = max(a_acc, 0.0) * float(self.acc_range)
        brake = max(-a_acc, 0.0) * float(self.acc_range)
        steer_cmd = a_ste * float(self.steer_range)
        return throttle, steer_cmd, brake

    def _check_waypoint(self, location_or_list, thresh=10.0):
        if self.ego_transform is None:
            return False
        ego_loc = self.ego_transform.location
        if isinstance(location_or_list, carla.Location):
            dx = ego_loc.x - location_or_list.x
            dy = ego_loc.y - location_or_list.y
            return (dx * dx + dy * dy) ** 0.5 < thresh
        try:
            for item in location_or_list:
                if isinstance(item, carla.Location):
                    dx = ego_loc.x - item.x
                    dy = ego_loc.y - item.y
                else:
                    x = float(item[0])
                    y = float(item[1])
                    dx = ego_loc.x - x
                    dy = ego_loc.y - y
                if (dx * dx + dy * dy) ** 0.5 < thresh:
                    return True
        except Exception:
            pass
        return False

    def _get_actor_polygons_tracked(self):
        actor_poly_dict = {}
        if not self._tracked_vehicle_ids:
            return actor_poly_dict

        tracked_set = set(int(v) for v in self._tracked_vehicle_ids)
        try:
            all_actors = self.world.get_actors(list(tracked_set))
        except Exception:
            return actor_poly_dict

        alive_ids = []
        for actor in all_actors:
            aid = int(actor.id)
            if aid not in tracked_set:
                continue
            try:
                if hasattr(actor, "is_alive") and (not actor.is_alive):
                    continue
            except Exception:
                pass

            try:
                poly = bbox_to_polygon_from_actor(actor)
                actor_poly_dict[aid] = poly
                alive_ids.append(aid)
            except Exception:
                continue

        self._tracked_vehicle_ids = alive_ids
        return actor_poly_dict

    def _create_ego_car(self, transform):
        try:
            ego_bp = self.world.get_blueprint_library().filter("model3")[0]
            ego_bp.set_attribute("color", "0, 0, 0")
            ego_bp.set_attribute("role_name", "hero")
        except Exception:
            return False

        try:
            self.ego_car = self.world.spawn_actor(ego_bp, transform)
            return True
        except Exception:
            self.ego_car = None
            return False

    def _create_others_vehicles(self, vehicle_spawn_points):
        vehicles_id_list = []
        try:
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_hybrid_physics_mode(True)

            blueprints = self.world.get_blueprint_library().filter("vehicle.*")
            blueprint_library = [
                x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
            ]
            blueprint_library = sorted(blueprint_library, key=lambda bp: bp.id)

            number_of_spawn_points = len(vehicle_spawn_points)
            if self.number_of_vehicles >= number_of_spawn_points:
                logging.warning(
                    "requested %d vehicles, but only %d spawn points",
                    self.number_of_vehicles,
                    number_of_spawn_points,
                )
                self.number_of_vehicles = max(0, number_of_spawn_points - 1)

            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor
            batch = []
            random.shuffle(vehicle_spawn_points)

            for n, transform in enumerate(vehicle_spawn_points):
                if n >= int(self.number_of_vehicles):
                    break
                blueprint = random.choice(blueprint_library)
                if blueprint.has_attribute("color"):
                    color = random.choice(blueprint.get_attribute("color").recommended_values)
                    blueprint.set_attribute("color", color)
                if blueprint.has_attribute("driver_id"):
                    driver_id = random.choice(
                        blueprint.get_attribute("driver_id").recommended_values
                    )
                    blueprint.set_attribute("driver_id", driver_id)
                blueprint.set_attribute("role_name", "autopilot")
                batch.append(
                    SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, 8000))
                )

            for response in self.client.apply_batch_sync(batch, self.settings.synchronous_mode):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_id_list.append(int(response.actor_id))
        except Exception:
            pass
        return vehicles_id_list

    def _get_collision(self, event):
        try:
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            if intensity > 0:
                self.collision = True
        except Exception:
            self.collision = True

    def _get_semantic_image(self, data):
        try:
            data.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.semantic_data = array
        except Exception:
            pass

    def _create_sensors(self):
        try:
            sem = self.sensors.create_semantic_cam(self.ego_car, self.img_size, use_queue=True)
            if isinstance(sem, tuple) and len(sem) == 2:
                self.semantic_sensor, self.semantic_queue = sem
            else:
                self.semantic_sensor = sem
                self.semantic_queue = None
                if self.semantic_sensor is not None:
                    self.semantic_sensor.listen(lambda data: self._get_semantic_image(data))
        except Exception:
            self.semantic_sensor = None
            self.semantic_queue = None

        try:
            self.collision_sensor = self.sensors.create_collision(self.ego_car)
            self.collision = False
            if self.collision_sensor is not None:
                self.collision_sensor.listen(lambda event: self._get_collision(event))
        except Exception:
            self.collision_sensor = None

    def _destroy_sensors(self):
        if self.collision_sensor is not None:
            safe_call(self.collision_sensor.stop)
            safe_call(self.collision_sensor.destroy)
            self.collision_sensor = None

        if self.semantic_sensor is not None:
            safe_call(self.semantic_sensor.stop)
            safe_call(self.semantic_sensor.destroy)
            self.semantic_sensor = None

        self.semantic_queue = None
        gc.collect()

    def _destroy_all_actors(self):
        self._destroy_sensors()
        try:
            for actor_filter in ["vehicle.*", "controller.ai.walker", "walker.*"]:
                for actor in self.world.get_actors().filter(actor_filter):
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except Exception:
                        pass
            self._tick()
        except Exception:
            pass

    def _tick(self):
        if self.settings.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _set_mode(self, synchronous=False):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def reset_carla(self):
        """Clean up CARLA resources and restore async mode."""
        try:
            if self.render_enabled:
                safe_call(pygame.quit)
        except Exception:
            pass

        try:
            if self._gm_renderer is not None:
                safe_call(self._gm_renderer.close)
        except Exception:
            pass

        try:
            self.settings.no_rendering_mode = False
            self._set_mode(False)
            self._destroy_all_actors()
        except Exception:
            pass
        print("Carla client restored.")
