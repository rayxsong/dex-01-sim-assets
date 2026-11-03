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

"""Flat 32 x 32 tactile pad with a 0.5 kg gear dropped onto it (SuperDex port of simple_example.py).

The settled taxel forces sum to the gear's weight, 0.5 kg x 9.81 m/s^2 = 4.905 N.
"""

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import superdex.physics as sdp
import trimesh
from dex01_common.patterns import grid_pattern
from dex01_superdex import TactileSensor
from dex01_superdex.scenes import (
    ENVIRONMENT_LAYER,
    TIME_STEP_S,
    add_ground,
    add_object,
    create_scene,
    gear_mesh,
    mesh_shape,
    set_pose,
    use_tactile_contact,
)
from dex01_superdex.tactile_sensor import matrix_to_transform

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--headless", action="store_true", help="Run without the Rerun viewer.")
parser.add_argument(
    "--max-steps", type=int, default=0, help="Stop after this many physics steps (0 runs until Ctrl+C)."
)
parser.add_argument("--save", type=Path, help="Also record the run to this .rrd file.")
args = parser.parse_args()

SENSOR_SIZE = (0.2, 0.1, 0.01)
PAD_HEIGHT_M = 0.05
GRID = (32, 32)
RESOLUTION = (SENSOR_SIZE[0] / GRID[0], SENSOR_SIZE[1] / GRID[1])
GEAR_MASS_KG = 0.5
DROP_HEIGHT_M = 0.03
RESET_PERIOD_S = 100 / 120  # same cadence as the Isaac Lab example: 100 steps at 120 Hz
REPORT_PERIOD_S = 0.5
DRAW_PERIOD_S = 1 / 30


def gear_start_pose():
    pose = np.eye(4)
    pose[2, 3] = PAD_HEIGHT_M + SENSOR_SIZE[2] / 2 + DROP_HEIGHT_M + gear_mesh().extents[2] / 2
    return pose


def main():
    sdp.initialize(num_worker_threads=0)
    scene = create_scene("dex01_flat_pad")
    add_ground(scene)
    pad_shape = mesh_shape(trimesh.creation.box(SENSOR_SIZE))
    pad = scene.create_rigid_actor(
        name="pad",
        layer=ENVIRONMENT_LAYER,
        shape=pad_shape,
        is_static=True,
        world_from_local=matrix_to_transform(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, PAD_HEIGHT_M], [0, 0, 0, 1.0]])
        ),
        collider_type=sdp.ColliderType.MESH,
    )
    sdp.release_shape(pad_shape)
    use_tactile_contact(pad)
    gear = add_object(scene, "gear", gear_mesh(), GEAR_MASS_KG, gear_start_pose())
    pattern = grid_pattern(size=(SENSOR_SIZE[0] - RESOLUTION[0], SENSOR_SIZE[1] - RESOLUTION[1]), resolution=RESOLUTION)
    sensor = TactileSensor(pad, gear, pattern, offset_pos=(0, 0, SENSOR_SIZE[2] / 2))
    print(f"Tactile sensor on 'pad': {sensor.num_taxels} taxels, frame {pattern.frame_shape}, counter 'gear'")

    view = None
    if not args.headless or args.save:
        from dex01_superdex.viewer import TactileRerunView

        view = TactileRerunView(
            scene, [sensor], ["pad"], spawn=not args.headless, save_path=args.save, taxel_radius=0.0012
        )

    reset_steps = round(RESET_PERIOD_S / TIME_STEP_S)
    report_steps = round(REPORT_PERIOD_S / TIME_STEP_S)
    draw_steps = max(1, round(DRAW_PERIOD_S / TIME_STEP_S))
    step = 0
    wall_start = time.perf_counter()
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C ends an open-ended run
        while not args.max_steps or step < args.max_steps:
            if step and step % reset_steps == 0:
                set_pose(gear, gear_start_pose())
                print("reset")
            scene.step(TIME_STEP_S)
            sensor.update()
            step += 1
            if step % report_steps == 0:
                forces = sensor.data.normal_forces
                print(f"peak taxel force {forces.max():7.4f} N   total {forces.sum():7.4f} N")
            if view is not None and step % draw_steps == 0:
                view.log()
                lag = step * TIME_STEP_S - (time.perf_counter() - wall_start)
                if not args.headless and lag > 0:
                    time.sleep(lag)  # pace to real time while watching
    if view is not None:
        view.close()
    sensor.close()
    sdp.destroy_scene(scene)
    sdp.shutdown()


if __name__ == "__main__":
    main()
