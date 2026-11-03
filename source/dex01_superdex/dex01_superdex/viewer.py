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

"""Rerun view of a SuperDex scene with live DEX-01 tactile data.

One window shows the scene (SuperDex's own Rerun logger), taxels as points coloured by force,
the firmware-mapped frame of each sensor and a plot of each sensor's total normal force. A hand is
drawn with its USD visual meshes and colours instead of SuperDex's collision shapes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .scenes import HAND_LAYER

UNUSED_CELL_RGB = (140, 140, 140)
FRAME_UPSCALE = 8  # pixels per taxel cell in the logged frames


class TactileRerunView:
    def __init__(self, scene, sensors, labels, spawn=True, save_path=None, taxel_radius=0.0003, hand=None):
        import matplotlib
        import rerun as rr
        import rerun.blueprint as rrb
        from superdex.physics.rerun.logger import RerunLogger
        from superdex.physics.rerun.logger_cfg import RerunLoggerCfg

        # rr.spawn() finds the viewer on PATH; include the environment's bin directory so it works
        # without activating the virtual environment.
        bin_dir = os.path.dirname(sys.executable)
        if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        self._rr = rr
        self.sensors, self.labels = sensors, labels
        self.taxel_radius = taxel_radius
        self.hand = hand
        self._cmap = matplotlib.colormaps["inferno"]
        # Scene data is Z-up; log it unconverted and declare the convention to the viewer instead.
        cfg = RerunLoggerCfg(
            application_id="dex01_superdex", spawn=spawn, save_path=save_path, coordinate_system="polyscope"
        )
        self.logger = RerunLogger(cfg)
        self.logger.reset_scene(scene)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        scene_contents = ["+ world/**"]
        if hand is not None:
            # SuperDex's logger files actors under world/<contact layer>/; hide the hand's collision shapes.
            scene_contents.append(f"- world/{HAND_LAYER}/{hand.actor.get_name()}/**")
            for link, meshes in hand.visuals.items():
                for k, mesh in enumerate(meshes):
                    rr.log(
                        f"world/hand_visual/{link}/{k}",
                        rr.Mesh3D(
                            vertex_positions=mesh.points,
                            triangle_indices=mesh.faces,
                            albedo_factor=[int(255 * c) for c in mesh.rgba],
                        ),
                        static=True,
                    )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(origin="world", contents=scene_contents, name="Scene"),
                        rrb.TimeSeriesView(origin="force", name="Total normal force (N)"),
                        column_shares=[3, 2],
                    ),
                    rrb.Horizontal(*[rrb.Spatial2DView(origin=f"frame/{label}", name=label) for label in labels]),
                    row_shares=[3, 1],
                ),
                collapse_panels=True,
            )
        )

    def log(self):
        rr = self._rr
        self.logger.log_frame()
        if self.hand is not None:
            for name, link in self.hand.links.items():
                pose = link.get_root_transform()
                rr.log(
                    f"world/hand_visual/{name}",
                    rr.Transform3D(translation=list(pose.translation), quaternion=pose.rotation.tolist()),
                )
        frames = [sensor.tactile_image() for sensor in self.sensors]
        peak = max(1e-6, max(frame.max() for frame in frames))  # one shared scale per frame
        points, colors = [], []
        for sensor, label, frame in zip(self.sensors, self.labels, frames):
            forces = sensor.data.normal_forces
            points.append(sensor.data.taxel_points_w)
            rgb = (self._cmap(forces / peak)[:, :3] * 255).astype(np.uint8)
            rgb[forces <= 0] = (110, 110, 120)
            colors.append(rgb)
            image = (self._cmap(frame / peak)[:, :, :3] * 255).astype(np.uint8)
            image[~sensor.frame_mask()] = UNUSED_CELL_RGB
            image = np.repeat(np.repeat(image, FRAME_UPSCALE, axis=0), FRAME_UPSCALE, axis=1)
            rr.log(f"frame/{label}", rr.Image(image))
            rr.log(f"force/{label}", rr.Scalars(float(forces.sum())))
        rr.log(
            "world/taxels",
            rr.Points3D(np.concatenate(points), colors=np.concatenate(colors), radii=self.taxel_radius),
        )

    def close(self):
        self.logger.close()
