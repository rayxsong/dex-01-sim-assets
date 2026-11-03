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

"""Scene pieces shared by the SuperDex examples. Scenes are Z-up with gravity along -Z."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import superdex.physics as sdp
import trimesh
from dex01_common.usd_hand import HandLink, HandMesh, load_dg5f_dex01
from scipy.spatial import ConvexHull

from .tactile_sensor import REAL, matrix_to_transform

GRAVITY_M_S2 = 9.81
TIME_STEP_S = 1.0 / 240.0
# The default Newton tolerance (1e-3) leaves settled contact forces about 0.02% short of the load;
# 1e-5 brings the known-load checks within 0.001%.
NEWTON_ABS_TOL = 1e-5

# Spur gear outline; proportions approximate a medium Factory gear (NVIDIA's USD gear is not
# redistributable). SuperDex collides against its SDF, so the teeth are real non-convex geometry.
GEAR_ROOT_RADIUS_M = 0.02
GEAR_TOOTH_HEIGHT_M = 0.005
GEAR_TEETH = 16
GEAR_THICKNESS_M = 0.01

# Contact that acts on penetration, like PhysX with rest_offset = 0. SuperDex's default penalty
# switches on 1 mm before touching and ramps over 10 mm, so a light object settles in a gap where
# no taxel penetrates the counter SDF; the depth model then has nothing to distribute.
PENALTY_THRESHOLD_M = 0.0
PENALTY_SMOOTHING_HALF_DISTANCE_M = 5e-5
# Penalty contact is stiff (1e9 Pa/m); normal damping keeps servo-driven presses from bouncing.
# It changes transients only, not settled loads.
NORMAL_DAMPING_S_M = 10.0

HAND_LAYER = "hand"
OBJECT_LAYER = "object"
ENVIRONMENT_LAYER = "environment"


def use_tactile_contact(actor: sdp.Actor):
    """Apply the penetration-based contact settings; needed on both actors of a sensed contact."""
    params = actor.get_contact_params()
    params.penalty_threshold_default = PENALTY_THRESHOLD_M
    params.penalty_smoothing_half_distance = PENALTY_SMOOTHING_HALF_DISTANCE_M
    params.normal_viscous_damping_coefficient = NORMAL_DAMPING_S_M
    actor.set_contact_params(params)


def create_scene(name: str) -> sdp.Scene:
    scene = sdp.create_scene(name)
    scene.set_gravity([0.0, 0.0, -GRAVITY_M_S2])
    params = scene.get_solver_params()
    params.non_linear_solver.abs_tol = NEWTON_ABS_TOL
    scene.set_solver_params(params)
    return scene


def mesh_shape(mesh: trimesh.Trimesh) -> sdp.ShapeHandle:
    return sdp.create_tri_mesh_shape(
        coordinates=np.asarray(mesh.vertices, dtype=REAL).ravel(),
        connectivity=np.asarray(mesh.faces, dtype=np.int32).ravel(),
    )


def gear_mesh(scale: float = 1.0) -> trimesh.Trimesh:
    """Extruded spur gear centred on its origin; the underside is at -thickness/2."""
    from shapely.geometry import Polygon

    outline = []
    for k in range(GEAR_TEETH * 4):  # tooth top over two of every four outline samples
        angle = 2 * math.pi * k / (GEAR_TEETH * 4)
        radius = GEAR_ROOT_RADIUS_M + (GEAR_TOOTH_HEIGHT_M if k % 4 in (1, 2) else 0.0)
        outline.append((radius * math.cos(angle), radius * math.sin(angle)))
    mesh = trimesh.creation.extrude_polygon(Polygon(outline), GEAR_THICKNESS_M)
    mesh.apply_translation([0, 0, -GEAR_THICKNESS_M / 2])
    mesh.apply_scale(scale)
    return mesh


def add_ground(scene: sdp.Scene) -> sdp.Actor:
    plane = sdp.create_plane_shape(normal=[0, 0, 1], distance=0.0)
    ground = scene.create_rigid_actor(name="ground", layer=ENVIRONMENT_LAYER, shape=plane, is_static=True)
    sdp.release_shape(plane)
    return ground


def add_object(
    scene: sdp.Scene,
    name: str,
    mesh: trimesh.Trimesh,
    mass: float,
    world_from_local: np.ndarray,
    fixed: bool = False,
) -> sdp.Actor:
    """Rigid object with an SDF collider, so it can be a tactile counter object.

    A ``fixed`` object is a dynamic actor pinned by boundary conditions on all six DoFs. A static
    actor would not do: it has no contact sample points, so it never collides with the SDF-collider
    DEX-01 shell.
    """
    shape = mesh_shape(mesh)
    actor = scene.create_rigid_actor(
        name=name,
        layer=OBJECT_LAYER,
        shape=shape,
        is_static=False,
        mass=mass,
        world_from_local=matrix_to_transform(world_from_local),
        collider_type=sdp.ColliderType.SDF,
    )
    sdp.release_shape(shape)
    use_tactile_contact(actor)
    if fixed:
        dofs = np.arange(6, dtype=np.int32)
        values = np.zeros(6, dtype=REAL)
        actor.get_dof_values(dof_indices=dofs, out_dof_values=values)
        actor.add_boundary_condition_dofs_world_permanent(dof_indices=dofs, dof_values_world=values)
    return actor


def set_pose(actor: sdp.Actor, world_from_local: np.ndarray):
    actor.set_root_transform(matrix_to_transform(world_from_local))
    actor.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


@dataclass
class Hand:
    actor: sdp.Actor  # the articulation
    links: dict[str, sdp.Actor]  # nested link actors by link name
    dof_names: list[str]  # joint name of each DoF, in articulation order
    dof_limits: np.ndarray  # (num_dofs, 2) [rad]
    visuals: dict[str, list[HandMesh]]  # rendered meshes (link frame, USD colours) by link name

    def dof(self, joint_name: str) -> int:
        return self.dof_names.index(joint_name)

    def pose(self) -> np.ndarray:
        pose = np.zeros(len(self.dof_names), dtype=REAL)
        self.actor.get_articulated_pose(out_pose=pose)
        return pose.astype(float)

    def set_target(self, targets: np.ndarray):
        self.actor.set_articulated_target_pose(pose=np.asarray(targets, dtype=REAL))

    def set_pose(self, pose: np.ndarray):
        pose = np.asarray(pose, dtype=REAL)
        self.actor.set_articulated_pose_from_joints(pose=pose)
        self.actor.set_articulated_joint_velocities(velocities=np.zeros_like(pose))
        self.actor.reset_articulated_target_pose(pose=pose)


def _collision_mesh(mesh) -> trimesh.Trimesh:
    """The geometry PhysX collides with: convex approximations become hulls, SDF meshes stay exact."""
    if mesh.approximation not in ("convexHull", "convexDecomposition"):
        return trimesh.Trimesh(mesh.points, mesh.faces, process=False)
    # Build the hull from qhull's own vertex indices: re-merging its output (as trimesh's
    # convex_hull does) can leave near-coplanar hulls with open seams.
    hull = ConvexHull(mesh.points)
    used = np.unique(hull.simplices)
    remap = np.full(len(mesh.points), -1)
    remap[used] = np.arange(len(used))
    shape = trimesh.Trimesh(mesh.points[used], remap[hull.simplices], process=False)
    shape.fix_normals()
    return shape


def _link_mesh(link: HandLink) -> tuple[trimesh.Trimesh, sdp.ColliderType]:
    """One mesh per link: its collision meshes merged, or its visual meshes without a collider."""
    collisions = [_collision_mesh(m) for m in link.meshes if m.collision]
    if collisions:
        # Meshes PhysX collides through an SDF (the DEX-01 shell) get SuperDex's SDF collider too.
        sdf = any(m.approximation == "sdf" for m in link.meshes if m.collision)
        collider = sdp.ColliderType.SDF if sdf else sdp.ColliderType.MESH
        return trimesh.util.concatenate(collisions), collider
    visuals = [trimesh.Trimesh(m.points, m.faces, process=False) for m in link.meshes]
    return trimesh.util.concatenate(visuals), sdp.ColliderType.NONE


def add_hand(
    scene: sdp.Scene,
    world_from_root: np.ndarray,
    stiffness: float = 200.0,
    damping: float = 5.0,
    armature: float = 0.01,
) -> Hand:
    """DG-5F + DEX-01 hand, welded to the world, with joint-space position control.

    ``stiffness`` matches the Isaac Lab demo. Its ``damping = 80`` targets PhysX's implicit drive; with
    the joint ``armature`` used here it would lag targets by about 0.4 s, so ``damping`` is lower.
    Links ignore gravity, as in the Isaac Lab demo.
    """
    links = load_dg5f_dex01()
    joints, link_params = [], []
    shapes = {}  # identical link meshes (the five DEX-01 shells) share one shape and its SDF
    for link in links:
        mesh, collider = _link_mesh(link)
        key = (mesh.vertices.tobytes(), mesh.faces.tobytes())
        if key not in shapes:
            shapes[key] = mesh_shape(mesh)
        shape = shapes[key]
        joint = link.joint
        if link.parent < 0:
            joints.append(sdp.ArticulatedJointParams(name="world_weld", type=sdp.ArticulatedJointType.HARD))
            parent_from_joint, joint_from_link = np.eye(4), np.eye(4)
        else:
            parent_from_joint = link.pose_in_parent @ joint.pose1
            joint_from_link = np.linalg.inv(joint.pose1)
            if joint.axis is None:
                joints.append(
                    sdp.ArticulatedJointParams(
                        name=f"{link.name}_fixed",  # the USD names every DEX-01 mount "attach_joint"
                        type=sdp.ArticulatedJointType.HARD,
                        parent_link_from_joint=matrix_to_transform(parent_from_joint),
                    )
                )
            else:
                axis = joint.pose1[:3, :3].T @ joint.axis  # the axis in the joint frame
                joints.append(
                    sdp.ArticulatedJointParams(
                        name=joint.name,
                        type=sdp.ArticulatedJointType.REVOLUTE,
                        axis=[float(v) for v in axis],
                        parent_link_from_joint=matrix_to_transform(parent_from_joint),
                        min_limit=[float(v) for v in joint.range[0] * axis],
                        max_limit=[float(v) for v in joint.range[1] * axis],
                        inertia=armature,
                    )
                )
        link_params.append(
            sdp.ArticulatedLinkParams(
                name=link.name,
                parent_link=link.parent,
                parent_joint_from_link=matrix_to_transform(joint_from_link),
                shape=shape,
                collider_type=collider,
                layer=HAND_LAYER,
                mass=link.mass,
                has_gravity=False,
            )
        )
    params = sdp.ArticulatedActorParams(name="dg5f_right_dex01")
    params.world_from_root = matrix_to_transform(world_from_root)
    params.joints = joints
    params.links = link_params
    articulation = scene.create_articulated_actor(params)
    for shape in shapes.values():
        sdp.release_shape(shape)
    scene.enable_layer_contact_symmetric(HAND_LAYER, HAND_LAYER, enable=False)
    scene.enable_layer_contact_symmetric(HAND_LAYER, ENVIRONMENT_LAYER, enable=False)

    controller = sdp.PoseControllerParams(len(links))
    for index, link in enumerate(links):
        if link.joint is not None and link.joint.axis is not None:
            controller.joint_tracking[index] = sdp.PoseTrackingParams(stiffness=stiffness, damping=damping)
    articulation.add_articulated_pose_controller(controller)

    nested = {}
    for handle in articulation.get_nested_link_actors():
        actor = scene.get_actor(handle)
        nested[actor.get_name()] = actor
        if actor.get_collider_type() != sdp.ColliderType.NONE:
            use_tactile_contact(actor)
    revolute = [link for link in links if link.joint is not None and link.joint.axis is not None]
    if articulation.get_num_dofs() != len(revolute):
        raise RuntimeError(f"Expected {len(revolute)} DoFs, articulation has {articulation.get_num_dofs()}")
    hand = Hand(
        actor=articulation,
        links={link.name: nested[_nested_name(nested, link.name)] for link in links},
        dof_names=[link.joint.name for link in revolute],
        dof_limits=np.array([link.joint.range for link in revolute]),
        visuals={link.name: [m for m in link.meshes if not m.collision] for link in links},
    )
    hand.set_pose(np.zeros(len(revolute)))
    return hand


def _nested_name(nested: dict[str, sdp.Actor], link_name: str) -> str:
    """Nested link actors may be named after the link alone or prefixed with the articulation."""
    if link_name in nested:
        return link_name
    matches = [name for name in nested if name.endswith(link_name)]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot find the link actor for {link_name} among {sorted(nested)}")
    return matches[0]
