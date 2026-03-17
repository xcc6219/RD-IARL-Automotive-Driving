# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Auxiliary functions for the CARLA agents package."""

import math
import numpy as np
import carla


def draw_waypoints(world, waypoints, z=0.5):
    """Draw a list of waypoints in the CARLA world for debugging."""
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """Compute speed of a vehicle in km/h."""
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """Check if a target is within a distance ahead of the reference."""
    target_vector = np.array([
        target_transform.location.x - current_transform.location.x,
        target_transform.location.y - current_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return True
    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(
        np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def is_within_distance(target_location, current_location, orientation,
                        max_distance, d_angle_th_up, d_angle_th_low=0):
    """Check if a target is within a distance and angle range."""
    target_vector = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return True
    if norm_target > max_distance:
        return False

    forward_vector = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    d_angle = math.degrees(math.acos(np.clip(
        np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """Compute distance and angle between two locations."""
    target_vector = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])
    norm_target = np.linalg.norm(target_vector)
    if norm_target < 1e-6:
        return (0.0, 0.0)

    forward_vector = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    d_angle = math.degrees(math.acos(np.clip(
        np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """Compute 2D distance from a waypoint to a vehicle."""
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y
    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """Return the unit vector from location_1 to location_2."""
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """Euclidean distance between two 3D locations."""
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    return np.linalg.norm([x, y, z]) + np.finfo(float).eps


def positive(num):
    """Return the number if positive, else 0."""
    return num if num > 0.0 else 0.0
