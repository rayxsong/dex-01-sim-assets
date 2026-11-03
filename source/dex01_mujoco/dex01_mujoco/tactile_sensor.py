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

"""MuJoCo port of the penetration-based tactile sensor.

Same model as ``dex01_sim_asset.TactileSensor``: taxel points are tested against the counter
object's signed distance field, and the measured net contact force between the sensor body and
that object is resolved along the depth-weighted mean taxel normal and redistributed over the
penetrating taxels in proportion to their depth (TacSL). Summing taxel forces recovers the net
normal load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from dex01_common.patterns import TaxelPattern

from .sdf import body_sdf


def quat_to_mat(wxyz) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(wxyz, dtype=float))
    return mat.reshape(3, 3)


@dataclass
class TactileSensorData:
    taxel_points_w: np.ndarray
    taxel_normals_w: np.ndarray
    normal_forces: np.ndarray
    depths: np.ndarray
    depth_dots: np.ndarray
    net_force_w: np.ndarray = field(default_factory=lambda: np.zeros(3))


class TactileSensor:
    """Tactile sensor on one MuJoCo body, measuring contact against one counter body."""

    def __init__(
        self,
        model: mujoco.MjModel,
        sensor_body: str,
        counter_body: str,
        pattern: TaxelPattern,
        offset_pos=(0.0, 0.0, 0.0),
        offset_quat=(1.0, 0.0, 0.0, 0.0),
    ):
        self.model = model
        self.sensor_body = model.body(sensor_body).id
        self.counter_body = model.body(counter_body).id
        self.pattern = pattern
        rotation = quat_to_mat(offset_quat)
        self.taxel_points_l = pattern.points @ rotation.T + np.asarray(offset_pos)
        self.taxel_normals_l = pattern.axes[:, :, 2] @ rotation.T
        self.taxel2pixel = pattern.taxel2pixel
        self.num_taxels = len(self.taxel_points_l)
        self.counter_geoms = [
            g for g in range(model.ngeom) if model.geom_bodyid[g] == self.counter_body and self._collides(g)
        ]
        if not self.counter_geoms:
            raise RuntimeError(f"Counter body '{counter_body}' has no collision geoms")
        self._counter_radius = max(np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g] for g in self.counter_geoms)
        n = self.num_taxels
        self._data = TactileSensorData(np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n))

    def _collides(self, geom):
        return self.model.geom_contype[geom] != 0 or self.model.geom_conaffinity[geom] != 0

    @property
    def data(self) -> TactileSensorData:
        return self._data

    def contact_force(self, data: mujoco.MjData) -> np.ndarray:
        """Net world-frame force the counter body exerts on the sensor body."""
        total, wrench = np.zeros(3), np.zeros(6)
        if data.ncon == 0:
            return total
        bodies = self.model.geom_bodyid[data.contact.geom[: data.ncon]]
        sensor_second = (bodies[:, 0] == self.counter_body) & (bodies[:, 1] == self.sensor_body)
        sensor_first = (bodies[:, 0] == self.sensor_body) & (bodies[:, 1] == self.counter_body)
        for i in np.flatnonzero(sensor_first | sensor_second):
            mujoco.mj_contactForce(self.model, data, i, wrench)
            # The contact frame's first row is the normal from geom1 to geom2; the force acts on geom2.
            force_w = data.contact.frame[i].reshape(3, 3).T @ wrench[:3]
            total += force_w if sensor_second[i] else -force_w
        return total

    def _point_velocity(self, data, body, points):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return velocity[3:] + np.cross(velocity[:3], points - data.xpos[body])

    def update(self, data: mujoco.MjData) -> TactileSensorData:
        rotation = data.xmat[self.sensor_body].reshape(3, 3)
        points_w = self.taxel_points_l @ rotation.T + data.xpos[self.sensor_body]
        normals_w = self.taxel_normals_l @ rotation.T
        net_force_w = self.contact_force(data)

        depths = np.zeros(self.num_taxels)
        depth_dots = np.zeros(self.num_taxels)
        if np.linalg.norm(net_force_w) > 0.0:
            # Only taxels inside the counter body's bounding sphere can penetrate it.
            near = np.flatnonzero(
                np.linalg.norm(points_w - data.xpos[self.counter_body], axis=-1) <= self._counter_radius
            )
            points = points_w[near]
            raw = body_sdf(self.model, data, self.counter_geoms, points)
            depths[near] = np.minimum(raw, 0.0)
            relative = self._point_velocity(data, self.sensor_body, points) - self._point_velocity(
                data, self.counter_body, points
            )
            # Symmetric difference of the SDF along relative motion (same 1 ms half-width as Isaac Lab).
            eps = 0.001
            forward = body_sdf(self.model, data, self.counter_geoms, points + eps * relative)
            backward = body_sdf(self.model, data, self.counter_geoms, points - eps * relative)
            depth_dots[near] = np.where(raw < 0.0, (forward - backward) / (2 * eps), 0.0)

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
