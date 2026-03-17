# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data access layer for GlobalRoutePlanner, fetching topology from CARLA.
"""

import numpy as np


class GlobalRoutePlannerDAO(object):
    """Fetches and preprocesses map topology data for the GlobalRoutePlanner."""

    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap

    def get_topology(self):
        """Retrieve and process the road topology into segment dictionaries."""
        topology = []
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            topology.append(seg_dict)
        return topology

    def get_waypoint(self, location):
        """Return the waypoint closest to the given location."""
        return self._wmap.get_waypoint(location)

    def get_resolution(self):
        """Return the sampling resolution."""
        return self._sampling_resolution
