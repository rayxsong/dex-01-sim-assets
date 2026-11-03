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

"""Finger 2 articulates against a fixed gear with the wrist fixed (SuperDex port of the stationary press).

Each cycle: unloaded, approach to an aligned pose just short of the gear, a force-seeking hold that
drives the measured load to --press-force-n while keeping the planar pad parallel to the gear face,
then retract and release. Only joint position targets move the finger.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import superdex.physics as sdp
from dex01_common.patterns import dex01_mount, dex01_pattern
from dex01_superdex import TactileSensor
from dex01_superdex.scenes import (
    TIME_STEP_S,
    add_ground,
    add_hand,
    add_object,
    create_scene,
    gear_mesh,
)
from dex01_superdex.tactile_sensor import transform_to_matrix
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--headless", action="store_true", help="Run without the Rerun viewer.")
parser.add_argument(
    "--max-steps", type=int, default=0, help="Stop after this many physics steps (0 runs until Ctrl+C)."
)
parser.add_argument("--save", type=Path, help="Also record the run to this .rrd file.")
parser.add_argument("--press-force-n", type=float, default=1.0, help="Measured normal-load target during the hold.")
parser.add_argument("--press-time-scale", type=float, default=0.25, help="Scale of the 10 s cycle (default 2.5 s).")
parser.add_argument("--verification-report", type=Path, help="Write per-step observations and checks as JSON.")
args = parser.parse_args()
if not 0 < args.press_force_n <= 2:
    parser.error("--press-force-n must be in (0, 2] for this fixture")
if not 0.1 <= args.press_time_scale <= 1:
    parser.error("--press-time-scale must be in [0.1, 1]")

HAND_POS = (0.2, 0.0, 0.05)
HAND_QUAT_WXYZ = (-0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
GEAR_SCALE = 0.15
PRESS_JOINTS = ("rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4")
# Moderate closure: about 22% of MCP range, 22% of PIP range and 10% of DIP range.
BEND_FRACTION = np.array([0.225, 0.225, 0.10])
APPROACH_CLEARANCE_M = 0.00002
MAX_PRESS_DISTANCE_M = 0.0006
# SuperDex's penalty contact is stiffer than PhysX's or MuJoCo's. Setting the targets to the current
# joints plus the full correction (as the Isaac Lab demo does) chatters against it, so the hold
# integrates a fraction of each correction into the previous targets, with a smaller per-step limit
# than Isaac Lab's 0.025 rad (~2 mm of fingertip travel).
MAX_CORRECTION_RAD = 0.002
CORRECTION_GAIN = 0.5
REPORT_PERIOD_S = 0.05
DRAW_PERIOD_S = 1 / 30


def press_motion(t):
    """Closure fraction and phase over the 10 s cycle; smoothstep keeps target velocity zero at boundaries."""
    t = t % 10.0
    if t < 1.0:
        return 0.0, "unloaded"
    if t < 4.0:
        u = min(1.0, (t - 1.0) / 2.5)
        return u * u * (3 - 2 * u), "approach"
    if t < 6.0:
        return 1.0, "hold"
    if t < 9.0:
        u = (t - 6.0) / 3.0
        return 1 - u * u * (3 - 2 * u), "retract"
    return 0.0, "unloaded"


def pose_matrix(pos, quat_wxyz):
    pose = np.eye(4)
    w, x, y, z = quat_wxyz
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = pos
    return pose


def frame_from_z(z):
    x = np.cross([0, 1.0, 0], z)
    if np.linalg.norm(x) < 0.1:
        x = np.cross([1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def patch_jacobian(link, dofs, point):
    """Linear and angular Jacobian of a point on a link for the given DoFs (world frame)."""
    jacobian = np.asarray(link.get_articulated_jacobian(), dtype=float).reshape(6, -1)[:, dofs]
    # SuperDex reports the link's centre-of-mass velocity; shift it to the patch point.
    com = np.asarray(link.get_center_of_mass_transform().translation, dtype=float)
    jacobian[:3] += np.cross(jacobian[3:].T, point - com).T
    return jacobian


def main():
    sdp.initialize(num_worker_threads=0)
    scene = create_scene("dex01_stationary_press")
    add_ground(scene)
    hand = add_hand(scene, pose_matrix(HAND_POS, HAND_QUAT_WXYZ))
    gear = gear_mesh(scale=GEAR_SCALE)
    parked = np.eye(4)
    parked[:3, 3] = (0.0, 0.0, 0.5)
    mount_pos, mount_quat = dex01_mount()
    dofs = [hand.dof(name) for name in PRESS_JOINTS]
    limits = hand.dof_limits[dofs]
    bend = limits[:, 1] * BEND_FRACTION
    link = hand.links["rl_dg_2_sensor"]

    # Setup-only pose sampling at the endpoint places the gear; the run itself uses joint targets only.
    endpoint = np.zeros(len(hand.dof_names))
    endpoint[dofs] = bend
    hand.set_pose(endpoint)
    sensor_pose = transform_to_matrix(link.get_root_transform())
    pattern = dex01_pattern()
    w, x, y, z = mount_quat
    mount_rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    points_l = pattern.points @ mount_rotation.T + mount_pos
    normals_l = pattern.axes[:, :, 2] @ mount_rotation.T
    # The planar pad: taxels sharing the most common normal (the rounded transition is excluded).
    shared = normals_l @ normals_l.T > 1 - 1e-6
    face_normal_l = normals_l[np.argmax(shared.sum(1))]
    flat = normals_l @ face_normal_l > 1 - 1e-6
    center = (points_l[flat] @ sensor_pose[:3, :3].T + sensor_pose[:3, 3]).mean(0)
    normal = sensor_pose[:3, :3] @ face_normal_l
    gear_pose = np.eye(4)
    gear_pose[:3, :3] = frame_from_z(normal)
    support = np.array([0, 0, gear.bounds[0, 2]])  # underside centre in the gear frame
    gear_pose[:3, 3] = center - gear_pose[:3, :3] @ support + normal * APPROACH_CLEARANCE_M
    gear_actor = add_object(scene, "gear", gear, 0.1, gear_pose, fixed=True)
    sensors = [
        TactileSensor(hand.links[f"rl_dg_{i}_sensor"], gear_actor, pattern, mount_pos, mount_quat) for i in range(1, 6)
    ]
    sensor = sensors[1]
    hand.set_pose(np.zeros(len(hand.dof_names)))
    wrist_pose = transform_to_matrix(hand.links["rl_dg_mount"].get_root_transform())
    print(f"Endpoint pad centre {np.round(center, 4)} m, normal {np.round(normal, 3)}, {flat.sum()} planar taxels")

    view = None
    if not args.headless or args.save:
        from dex01_superdex.viewer import TactileRerunView

        view = TactileRerunView(
            scene, sensors, [f"f{i}" for i in range(1, 6)], spawn=not args.headless, save_path=args.save, hand=hand
        )

    press_distance, integral, hold_joints = 0.0, np.zeros(3), bend.copy()
    observations, failure = [], None
    hold_loads, unloaded_loads = [], []
    report_steps = max(1, round(REPORT_PERIOD_S / TIME_STEP_S))
    draw_steps = max(1, round(DRAW_PERIOD_S / TIME_STEP_S))
    step = 0
    wall_start = time.perf_counter()
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C ends an open-ended run
        while not args.max_steps or step < args.max_steps:
            t = step * TIME_STEP_S / args.press_time_scale
            fraction, phase = press_motion(t)
            if t % 10.0 < 1.0:
                press_distance, integral, hold_joints = 0.0, np.zeros(3), bend.copy()
            joints = hand.pose()[dofs]
            targets = fraction * (hold_joints if phase == "retract" else bend)
            if phase == "hold":
                load = float(sensor.data.normal_forces.sum())
                speed = 0.001 if load < 0.01 else 0.003 * (args.press_force_n - load)
                press_distance = min(
                    MAX_PRESS_DISTANCE_M, max(0.0, press_distance + TIME_STEP_S * np.clip(speed, -0.001, 0.001))
                )
                patch_center = sensor.data.taxel_points_w[flat].mean(0)
                patch_normal = sensor.data.taxel_normals_w[flat].mean(0)
                patch_normal /= np.linalg.norm(patch_normal)
                rotation_error = np.cross(patch_normal, normal)
                integral = np.clip(integral + 20 * TIME_STEP_S * rotation_error, -0.02, 0.02)
                # Solve translation and orientation together; bending every joint alone tilts the pad.
                error = np.concatenate([center + normal * press_distance - patch_center, 3 * rotation_error + integral])
                correction = np.linalg.lstsq(patch_jacobian(link, dofs, patch_center), error, rcond=None)[0]
                step_correction = CORRECTION_GAIN * np.clip(correction, -MAX_CORRECTION_RAD, MAX_CORRECTION_RAD)
                hold_joints = np.clip(hold_joints + step_correction, limits[:, 0], limits[:, 1])
                targets = hold_joints
            all_targets = np.zeros(len(hand.dof_names))
            all_targets[dofs] = targets
            hand.set_target(all_targets)
            scene.step(TIME_STEP_S)
            for s in sensors:
                s.update()
            step += 1

            forces = [float(s.data.normal_forces.sum()) for s in sensors]
            cycle = t % 10.0
            if np.abs(transform_to_matrix(hand.links["rl_dg_mount"].get_root_transform()) - wrist_pose).max() > 1e-5:
                failure = "Wrist moved during finger articulation"
            if np.abs(transform_to_matrix(gear_actor.get_root_transform()) - gear_pose).max() > 1e-5:
                failure = "Stationary object moved"
            if phase == "hold" and cycle >= 5.0:
                hold_loads.append(forces[1])
            if phase == "unloaded" and cycle >= 9.5:
                unloaded_loads.append(max(forces))
            if args.verification_report and step % 3 == 0:
                observations.append({
                    "step": step,
                    "phase": phase,
                    "joint_positions_rad": joints.tolist(),
                    "forces_n": forces,
                    "press_distance_m": press_distance,
                })
            if step % report_steps == 0:
                print(
                    f"{phase:9s} joints deg {np.round(np.degrees(hand.pose()[dofs]), 1)}  "
                    + "  ".join(f"f{i} {f:6.3f}N" for i, f in enumerate(forces, 1))
                )
            if view is not None and step % draw_steps == 0:
                view.log()
                lag = step * TIME_STEP_S - (time.perf_counter() - wall_start)
                if not args.headless and lag > 0:
                    time.sleep(lag)
            if failure:
                break
    if view is not None:
        view.close()

    summary = {
        "failure": failure,
        "target_force_n": args.press_force_n,
        "late_hold_mean_force_n": float(np.mean(hold_loads)) if hold_loads else None,
        "late_hold_min_force_n": float(np.min(hold_loads)) if hold_loads else None,
        "released_max_force_n": float(np.max(unloaded_loads)) if unloaded_loads else None,
    }
    print("\nSummary:", json.dumps(summary))
    if args.verification_report:
        args.verification_report.write_text(
            json.dumps({**summary, "dt_s": TIME_STEP_S, "observations": observations}) + "\n"
        )
    for s in sensors:
        s.close()
    sdp.destroy_scene(scene)
    sdp.shutdown()
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
