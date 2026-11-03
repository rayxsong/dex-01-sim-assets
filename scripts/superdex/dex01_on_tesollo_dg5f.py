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

"""DEX-01 fingertips on the Tesollo DG-5F hand in SuperDex (port of dex01_on_tesollo_dg5f.py).

A 0.1 kg gear rests on fingers 2-5 in turn. The loaded finger reports the known 0.981 N weight in
its firmware-mapped 32 x 32 frame; the other fingers read zero.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import superdex.physics as sdp
import trimesh
from dex01_common.patterns import dex01_mount, dex01_pattern
from dex01_superdex import TactileSensor
from dex01_superdex.scenes import (
    GRAVITY_M_S2,
    TIME_STEP_S,
    add_ground,
    add_hand,
    add_object,
    create_scene,
    gear_mesh,
    set_pose,
)
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--headless", action="store_true", help="Run without the Rerun viewer.")
parser.add_argument(
    "--max-steps", type=int, default=0, help="Stop after this many physics steps (0 runs until Ctrl+C)."
)
parser.add_argument("--save", type=Path, help="Also record the run to this .rrd file.")
parser.add_argument("--flat-indenter", action="store_true", help="Use a 6 mm cube instead of the gear.")
parser.add_argument("--verification-report", type=Path, help="Write per-step forces and settled-load checks as JSON.")
args = parser.parse_args()

# Demo parameters carried over from the Isaac Lab script.
INDENTER_MASS_KG = 0.1  # mass * gravity = 0.981 N
GEAR_DISPLAY_SCALE = 0.15  # keeps the gear footprint on the active pad
FLAT_INDENTER_SIZE_M = 0.006
HAND_POS = (0.2, 0.0, 0.05)
HAND_QUAT_WXYZ = (-0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
SETTLE_CLEARANCE_M = 0.0005
HORIZONTAL_NORMAL_MIN_Z = 0.995
PRESSED_FINGERS = (2, 3, 4, 5)
SECONDS_PER_FINGER = 1.25  # the final 20% is checked as settled load
REPORT_PERIOD_S = 1 / 12
DRAW_PERIOD_S = 1 / 30


def hand_pose():
    pose = np.eye(4)
    w, x, y, z = HAND_QUAT_WXYZ
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = HAND_POS
    return pose


def indenter_mesh():
    if args.flat_indenter:
        return trimesh.creation.box([FLAT_INDENTER_SIZE_M] * 3)
    return gear_mesh(scale=GEAR_DISPLAY_SCALE)


def rest_pose_on(sensor, half_height):
    """Indenter pose just above the finger's upward-facing patch."""
    horizontal = sensor.data.taxel_normals_w[:, 2] > HORIZONTAL_NORMAL_MIN_Z
    if not horizontal.any():
        raise RuntimeError("No upward-facing sensor patch for the gravity-load demonstration")
    pose = np.eye(4)
    pose[:3, 3] = sensor.data.taxel_points_w[horizontal].mean(0)
    pose[2, 3] = sensor.data.taxel_points_w[:, 2].max() + SETTLE_CLEARANCE_M + half_height
    return pose


def main():
    sdp.initialize(num_worker_threads=0)
    scene = create_scene("dex01_on_tesollo_dg5f")
    add_ground(scene)
    hand = add_hand(scene, hand_pose())
    mesh = indenter_mesh()
    parked = np.eye(4)
    parked[:3, 3] = (0.0, 0.3, 0.3)
    indenter = add_object(scene, "indenter", mesh, INDENTER_MASS_KG, parked)
    pattern = dex01_pattern()
    mount_pos, mount_quat = dex01_mount()
    sensors = [
        TactileSensor(hand.links[f"rl_dg_{i}_sensor"], indenter, pattern, mount_pos, mount_quat) for i in range(1, 6)
    ]
    for sensor in sensors:
        sensor.update()
    print(f"{len(sensors)} DEX-01 sensors x {sensors[0].num_taxels} taxels, frame {pattern.frame_shape}")

    labels = [f"f{i}" for i in range(1, 6)]
    view = None
    if not args.headless or args.save:
        from dex01_superdex.viewer import TactileRerunView

        view = TactileRerunView(scene, sensors, labels, spawn=not args.headless, save_path=args.save, hand=hand)

    steps_per_finger = round(SECONDS_PER_FINGER / TIME_STEP_S)
    report_steps = round(REPORT_PERIOD_S / TIME_STEP_S)
    draw_steps = max(1, round(DRAW_PERIOD_S / TIME_STEP_S))
    observations, settled = [], {f: [] for f in PRESSED_FINGERS}
    step = 0
    wall_start = time.perf_counter()
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C ends an open-ended run
        while not args.max_steps or step < args.max_steps:
            cycle_step = step % (steps_per_finger * len(PRESSED_FINGERS))
            finger = PRESSED_FINGERS[cycle_step // steps_per_finger]
            if cycle_step % steps_per_finger == 0:
                set_pose(indenter, rest_pose_on(sensors[finger - 1], mesh.extents[2] / 2))

            scene.step(TIME_STEP_S)
            for sensor in sensors:
                sensor.update()
            step += 1

            totals = [float(s.data.normal_forces.sum()) for s in sensors]
            if cycle_step % steps_per_finger >= 0.8 * steps_per_finger:
                settled[finger].append(totals)
            if args.verification_report:
                observations.append({"step": step, "target_finger": finger, "forces_n": totals})
            if step % report_steps == 0:
                print("  ".join(f"f{i} {t:7.3f}N" for i, t in enumerate(totals, 1)))
            if view is not None and step % draw_steps == 0:
                view.log()
                lag = step * TIME_STEP_S - (time.perf_counter() - wall_start)
                if not args.headless and lag > 0:
                    time.sleep(lag)
    if view is not None:
        view.close()

    expected = INDENTER_MASS_KG * GRAVITY_M_S2
    summary = {}
    for finger, rows in settled.items():
        if not rows:
            continue
        rows = np.asarray(rows)
        others = np.delete(rows, finger - 1, axis=1)
        summary[f"f{finger}"] = {
            "mean_load_n": float(rows[:, finger - 1].mean()),
            "error_pct": float(100 * (rows[:, finger - 1].mean() - expected) / expected),
            "max_other_finger_n": float(np.abs(others).max()),
        }
    print(f"\nSettled load (expected {expected:.3f} N):")
    for name, s in summary.items():
        print(
            f"  {name}: {s['mean_load_n']:.4f} N ({s['error_pct']:+.2f}%), "
            f"other fingers max {s['max_other_finger_n']:.4f} N"
        )
    if args.verification_report:
        report = {
            "dt_s": TIME_STEP_S,
            "indenter": "flat" if args.flat_indenter else "gear",
            "settled": summary,
            "observations": observations,
        }
        args.verification_report.write_text(json.dumps(report, indent=1) + "\n")
    for sensor in sensors:
        sensor.close()
    sdp.destroy_scene(scene)
    sdp.shutdown()


if __name__ == "__main__":
    main()
