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

"""Read the DG-5F USD articulation (with DEX-01 fingertips) into a simulator-neutral link tree.

The USD asset is a flat PhysX articulation: every link is a sibling under ``/Root`` and the tree is
encoded by joints with ``localPos0/localRot0`` (frame in body0) and ``localPos1/localRot1`` (frame
in body1). Each child is therefore placed at ``body0 * pose0 * pose1^-1`` and its joint sits at
``pose1`` in the child frame. Mesh points are baked into their link frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HAND_USD = REPO_ROOT / "assets/robots/tesollo/dg5f_right.usda"
DEFAULT_RGBA = (0.62, 0.63, 0.65, 1.0)  # meshes without a material, e.g. the distal fingertip_base
_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


@dataclass
class HandMesh:
    name: str  # unique; a visual part that reuses a collision mesh shares its name
    points: np.ndarray  # (N, 3) in the link frame
    faces: np.ndarray  # (M, 3) triangles
    collision: bool
    rgba: tuple[float, float, float, float] | None = None  # visual meshes only
    approximation: str | None = None  # USD physics:approximation of a collision mesh, e.g. "convexHull", "sdf"


@dataclass
class HandJoint:
    name: str
    pose1: np.ndarray  # 4x4 joint frame in the child link frame
    axis: np.ndarray | None = None  # unit axis in the child link frame; None for a fixed joint
    range: np.ndarray | None = None  # (lower, upper) [rad]


@dataclass
class HandLink:
    name: str
    parent: int  # index into the link list, -1 for the root
    pose_in_parent: np.ndarray  # 4x4 link frame in the parent link frame (all joints at zero)
    mass: float
    joint: HandJoint | None
    meshes: list[HandMesh] = field(default_factory=list)


def _quat_to_mat(wxyz):
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pose(pos, quat):
    T = np.eye(4)
    T[:3, :3] = _quat_to_mat([quat.GetReal(), *quat.GetImaginary()])
    T[:3, 3] = list(pos)
    return T


def _gf_to_np(m):
    # Gf matrices use row vectors (p' = p M); transpose to the column-vector convention.
    return np.array(m, dtype=float).T


def _material_rgba(material):
    """Diffuse colour of an OmniPBR or UsdPreviewSurface material, or None."""
    if not material:
        return None
    from pxr import Usd

    for shader in Usd.PrimRange(material.GetPrim()):
        for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
            attr = shader.GetAttribute(name)
            if attr and attr.Get() is not None:
                return (*attr.Get(), 1.0)
    return None


def _triangulate(mesh):
    """Fan-triangulate a UsdGeom.Mesh; returns points, triangles and each triangle's source polygon."""
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, polygon, offset = [], [], 0
    for face, count in enumerate(counts):
        for k in range(1, count - 1):
            faces.append((indices[offset], indices[offset + k], indices[offset + k + 1]))
            polygon.append(face)
        offset += count
    return points, np.asarray(faces), np.asarray(polygon)


def _material_parts(prim, polygon):
    """(triangle mask, rgba) per material subset, or one part for the whole mesh."""
    from pxr import UsdGeom, UsdShade

    parts = []
    for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
        material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
        parts.append((np.isin(polygon, subset.GetIndicesAttr().Get()), _material_rgba(material)))
    if not parts:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        parts = [(np.ones(len(polygon), bool), _material_rgba(material))]
    return parts


def _is_rendered(prim):
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible and imageable.ComputePurpose() in (
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
    )


def _is_collision(prim, body_prim):
    from pxr import UsdPhysics

    while prim and prim != body_prim:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
        prim = prim.GetParent()
    return False


def _approximation(prim, body_prim):
    """physics:approximation authored on the mesh or the collision prim above it."""
    from pxr import UsdPhysics

    while prim and prim != body_prim:
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            return str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        prim = prim.GetParent()
    return None


class _UsdHandReader:
    def __init__(self, stage):
        from pxr import UsdGeom

        self.stage = stage
        self.xform_cache = UsdGeom.XformCache()
        self.bodies = self._read_bodies()
        self.children, parents = self._read_joints()
        self.roots = [path for path in self.bodies if path not in parents]
        if len(self.roots) != 1:
            raise RuntimeError(f"Expected one articulation root, found {self.roots}")

    def _read_bodies(self):
        from pxr import UsdPhysics

        bodies = {}
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass_attr = prim.GetAttribute("physics:mass")
                bodies[str(prim.GetPath())] = {
                    "prim": prim,
                    "name": prim.GetName(),
                    "mass": float(mass_attr.Get()) if mass_attr and mass_attr.Get() else 0.01,
                    "authored": _gf_to_np(self.xform_cache.GetLocalToWorldTransform(prim)),
                }
        return bodies

    def _read_joints(self):
        from pxr import UsdPhysics

        children = {path: [] for path in self.bodies}
        parents = {}
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets0, targets1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not targets0 or not targets1:
                continue  # world anchor of the fixed wrist; the scene places the root link instead
            body0, body1 = str(targets0[0]), str(targets1[0])
            pose0 = _pose(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            pose1 = _pose(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            hand_joint = HandJoint(prim.GetName(), pose1)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute = UsdPhysics.RevoluteJoint(prim)
                hand_joint.axis = pose1[:3, :3] @ _AXES[revolute.GetAxisAttr().Get()]
                hand_joint.range = np.radians([revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()])
            children[body0].append((body1, pose0 @ np.linalg.inv(pose1), hand_joint))
            parents[body1] = body0
        return children, parents

    def _body_meshes(self, body):
        from pxr import Usd, UsdGeom, UsdPhysics

        prims = Usd.PrimRange(body["prim"], Usd.TraverseInstanceProxies())
        for prim in prims:
            if prim != body["prim"] and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prims.PruneChildren()
            elif prim.IsA(UsdGeom.Mesh):
                yield prim

    def meshes(self, body):
        """Collision and visual meshes of a body, baked into the body frame."""
        from pxr import UsdGeom

        body_inv = np.linalg.inv(body["authored"])
        meshes = []
        for prim in self._body_meshes(body):
            points, faces, polygon = _triangulate(UsdGeom.Mesh(prim))
            if len(points) < 4 or len(faces) == 0:
                continue
            to_body = body_inv @ _gf_to_np(self.xform_cache.GetLocalToWorldTransform(prim))
            points = points @ to_body[:3, :3].T + to_body[:3, 3]
            collision = _is_collision(prim, body["prim"])
            stem = f"{body['name']}_{'col' if collision else 'vis'}_{len(meshes)}"
            if collision:
                meshes.append(HandMesh(stem, points, faces, True, approximation=_approximation(prim, body["prim"])))
            # Collision meshes are usually guides or invisible, but some (the DEX-01 variant's distal
            # fingertip_base) are the link's only rendered geometry, so they are drawn as well.
            if not _is_rendered(prim):
                continue
            # Each material subset is its own visual mesh so the parts keep their colours.
            parts = _material_parts(prim, polygon)
            for k, (mask, rgba) in enumerate(parts):
                if mask.any():
                    name = stem if len(parts) == 1 else f"{stem}_{k}"
                    meshes.append(HandMesh(name, points, faces[mask], False, rgba or DEFAULT_RGBA))
        return meshes

    def links(self):
        """Depth-first link list; every parent precedes its children."""
        links = []

        def visit(path, parent, pose_in_parent, joint):
            body = self.bodies[path]
            index = len(links)
            links.append(HandLink(body["name"], parent, pose_in_parent, body["mass"], joint, self.meshes(body)))
            for child, child_pose, child_joint in self.children[path]:
                visit(child, index, child_pose, child_joint)

        visit(self.roots[0], -1, np.eye(4), None)
        return links


def load_dg5f_dex01(usd_path: Path = HAND_USD) -> list[HandLink]:
    """DG-5F right hand with DEX-01 fingertips (USD variant ``fingertip = dex01``) as a link tree."""
    from pxr import Usd

    if usd_path.stat().st_size < 10_000:
        raise RuntimeError(f"{usd_path} looks like a git-lfs pointer; run 'git lfs pull' first")
    stage = Usd.Stage.Open(str(usd_path))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("fingertip").SetVariantSelection("dex01")
    return _UsdHandReader(stage).links()
