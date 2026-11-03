# Copyright (c) 2026 Analog Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SuperDex Physics port of the penetration-based tactile sensor.

Same model as ``dex01_sim_asset.TactileSensor``: taxel points are tested against the counter
object's signed distance field, and the measured net contact force between the sensor actor and
that object is resolved along the depth-weighted mean taxel normal and redistributed over the
penetrating taxels in proportion to their depth (TacSL). Summing taxel forces recovers the net
normal load. The SDF is SuperDex's own collider query, so non-convex counter objects are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import superdex.physics as sdp
from dex01_common.patterns import TaxelPattern
from scipy.spatial.transform import Rotation

REAL = np.float64 if sdp.uses_double_precision() else np.float32


def transform_to_matrix(transform: sdp.TransformRT) -> np.ndarray:
    """4x4 world-from-local matrix of a SuperDex transform (quaternions are x, y, z, w)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform.rotation.tolist()).as_matrix()
    matrix[:3, 3] = np.asarray(transform.translation, dtype=float)
    return matrix


def matrix_to_transform(matrix: np.ndarray) -> sdp.TransformRT:
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return sdp.TransformRT(sdp.Quaternion(x, y, z, w), [float(v) for v in matrix[:3, 3]])


@dataclass
class TactileSensorData:
    taxel_points_w: np.ndarray
    taxel_normals_w: np.ndarray
    normal_forces: np.ndarray
    depths: np.ndarray
    depth_dots: np.ndarray
    net_force_w: np.ndarray = field(default_factory=lambda: np.zeros(3))


class TactileSensor:
    """Tactile sensor on one SuperDex actor (rigid actor or articulation link), against one counter actor.

    The sensor registers the contact-force query it needs, so construct it before stepping the scene.
    """

    def __init__(
        self,
        sensor_actor: sdp.Actor,
        counter_actor: sdp.Actor,
        pattern: TaxelPattern,
        offset_pos=(0.0, 0.0, 0.0),
        offset_quat=(1.0, 0.0, 0.0, 0.0),
    ):
        self.sensor_actor = sensor_actor
        self.counter_actor = counter_actor
        self.pattern = pattern
        w, x, y, z = offset_quat
        rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
        self.taxel_points_l = pattern.points @ rotation.T + np.asarray(offset_pos)
        self.taxel_normals_l = pattern.axes[:, :, 2] @ rotation.T
        self.taxel2pixel = pattern.taxel2pixel
        self.num_taxels = len(self.taxel_points_l)
        # Contact-force queries need an actor with contact sample points. A static or SDF-only
        # actor has none, so the force is then read on the counter object and negated.
        self._read_on_sensor = sensor_actor.is_query_supported(sdp.QueryType.TOTAL_CONTACT_FORCE)
        queried = sensor_actor if self._read_on_sensor else counter_actor
        if not queried.is_query_supported(sdp.QueryType.TOTAL_CONTACT_FORCE):
            raise RuntimeError("Neither the sensor nor the counter actor supports contact-force queries")
        self._query = queried.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
        self._queried = queried
        self._has_results = False
        self._distances = np.zeros(self.num_taxels, dtype=REAL)
        n = self.num_taxels
        self._data = TactileSensorData(np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n))

    @property
    def data(self) -> TactileSensorData:
        return self._data

    def close(self):
        self._queried.cancel_query(self._query)

    def contact_force(self) -> np.ndarray:
        """Net world-frame force the counter actor exerts on the sensor actor."""
        try:
            if self._read_on_sensor:
                force = self.sensor_actor.get_contact_force_from_actor_world(self.counter_actor)
            else:
                force = -np.asarray(self.counter_actor.get_contact_force_from_actor_world(self.sensor_actor))
        except sdp.Error:
            if self._has_results:
                raise
            return np.zeros(3)  # query results exist only after the first step following registration
        self._has_results = True
        return np.asarray(force, dtype=float)

    def _sdf(self, points_w):
        self.counter_actor.get_points_distance_to_surface(
            points_world=points_w.astype(REAL), out_distances=self._distances
        )
        return self._distances.astype(float)

    @staticmethod
    def _point_velocity(actor, points_w):
        if actor.is_static():
            return np.zeros_like(points_w)
        com = np.asarray(actor.get_center_of_mass_transform().translation, dtype=float)
        linear = np.asarray(actor.get_linear_velocity(), dtype=float)
        angular = np.asarray(actor.get_angular_velocity(), dtype=float)
        return linear + np.cross(angular, points_w - com)

    def update(self) -> TactileSensorData:
        pose = transform_to_matrix(self.sensor_actor.get_root_transform())
        points_w = self.taxel_points_l @ pose[:3, :3].T + pose[:3, 3]
        normals_w = self.taxel_normals_l @ pose[:3, :3].T
        net_force_w = self.contact_force()

        depths = np.zeros(self.num_taxels)
        depth_dots = np.zeros(self.num_taxels)
        if np.linalg.norm(net_force_w) > 0.0:
            raw = self._sdf(points_w)
            depths = np.minimum(raw, 0.0)
            relative = self._point_velocity(self.sensor_actor, points_w) - self._point_velocity(
                self.counter_actor, points_w
            )
            # Symmetric difference of the SDF along relative motion (same 1 ms half-width as Isaac Lab).
            eps = 0.001
            forward = self._sdf(points_w + eps * relative)
            backward = self._sdf(points_w - eps * relative)
            depth_dots = np.where(raw < 0.0, (forward - backward) / (2 * eps), 0.0)

        depth_sum = min(depths.sum(), -1e-8)
        contact_normal = (normals_w * depths[:, None]).sum(0) / depth_sum
        contact_normal /= max(np.linalg.norm(contact_normal), 1e-8)
        net_normal_force = max(-(net_force_w @ contact_normal), 0.0)
        forces = np.where(depths < 0.0, net_normal_force * depths / depth_sum, 0.0)

        self._data = TactileSensorData(points_w, normals_w, forces, depths, depth_dots, net_force_w)
        return self._data

    def tactile_image(self) -> np.ndarray:
        """Taxel forces scattered into the firmware frame; unused cells are zero."""
        image = np.zeros(self.pattern.frame_shape)
        image[self.taxel2pixel[:, 0], self.taxel2pixel[:, 1]] = self._data.normal_forces
        return image

    def frame_mask(self) -> np.ndarray:
        mask = np.zeros(self.pattern.frame_shape, dtype=bool)
        mask[self.taxel2pixel[:, 0], self.taxel2pixel[:, 1]] = True
        return mask
