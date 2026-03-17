# -*- coding: utf-8 -*-
"""
Sensor factory for creating CARLA sensors (semantic camera, collision).
"""

import logging
import carla
from queue import Queue


class SensorManager:
    """Manages creation of CARLA sensors attached to the ego vehicle."""

    def __init__(self, world):
        self.world = world

    def create_camera(self, car, image_size, fov=60, sensor_attributes=None, use_queue=False):
        """Create an RGB camera sensor attached to the given vehicle."""
        try:
            sensor_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            if not sensor_attributes:
                sensor_attributes = {}
            sensor_attributes.setdefault('image_size_x', str(image_size[0]))
            sensor_attributes.setdefault('image_size_y', str(image_size[1]))
            sensor_attributes.setdefault('fov', str(fov))

            fds = self.world.get_settings().fixed_delta_seconds
            if fds and 'sensor_tick' not in sensor_attributes:
                sensor_attributes['sensor_tick'] = str(fds)

            for attr, value in sensor_attributes.items():
                sensor_bp.set_attribute(attr, value)

            camera_transform = carla.Transform(carla.Location(x=1.13, z=1.50))
            sensor = self.world.spawn_actor(
                sensor_bp, camera_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid
            )

            q = Queue() if use_queue else None
            if use_queue:
                sensor.listen(q.put)
            return sensor if not use_queue else (sensor, q)
        except Exception as e:
            logging.error(f"Failed to create camera sensor: {e}")
            return None

    def create_semantic_cam(self, car, image_size, fov=60, sensor_attributes=None, use_queue=False):
        """Create a semantic segmentation camera sensor."""
        try:
            sensor_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            if not sensor_attributes:
                sensor_attributes = {}
            sensor_attributes.setdefault('image_size_x', str(image_size[0]))
            sensor_attributes.setdefault('image_size_y', str(image_size[1]))
            sensor_attributes.setdefault('fov', str(fov))

            fds = self.world.get_settings().fixed_delta_seconds
            if fds and 'sensor_tick' not in sensor_attributes:
                sensor_attributes['sensor_tick'] = str(fds)

            for attr, value in sensor_attributes.items():
                sensor_bp.set_attribute(attr, value)

            camera_transform = carla.Transform(carla.Location(x=1.13, z=1.50))
            sensor = self.world.spawn_actor(
                sensor_bp, camera_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid
            )

            q = Queue() if use_queue else None
            if use_queue:
                sensor.listen(q.put)
            return sensor if not use_queue else (sensor, q)
        except Exception as e:
            logging.error(f"Failed to create semantic camera sensor: {e}")
            return None

    def create_collision(self, car):
        """Create a collision sensor attached to the given vehicle."""
        try:
            sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            sensor = self.world.spawn_actor(
                sensor_bp, carla.Transform(), attach_to=car, attachment_type=carla.AttachmentType.Rigid
            )
            return sensor
        except Exception as e:
            logging.error(f"Failed to create collision sensor: {e}")
            return None
