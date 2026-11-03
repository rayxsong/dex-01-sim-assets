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

"""Write the DG-5F + DEX-01 hand as an MJCF body tree with STL meshes, cached by input hash."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from dex01_common import usd_hand
from dex01_common.usd_hand import HAND_USD, REPO_ROOT, HandLink, load_dg5f_dex01

OUTPUT_DIR = REPO_ROOT / "assets/robots/tesollo/mjcf"
HAND_XML = OUTPUT_DIR / "dg5f_right_dex01.xml"
_STAMP = OUTPUT_DIR / "build_stamp.json"


def _mat_to_quat(R):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _fmt(values):
    return " ".join(f"{v:.9g}" for v in values)


def _source_hash():
    digest = hashlib.sha256(HAND_USD.read_bytes())
    for source in (Path(__file__), Path(usd_hand.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _emit(links: list[HandLink], index: int, depth: int) -> list[str]:
    link = links[index]
    pad = "  " * depth
    pose = link.pose_in_parent
    lines = [
        f'{pad}<body name="{link.name}" pos="{_fmt(pose[:3, 3])}"'
        f' quat="{_fmt(_mat_to_quat(pose[:3, :3]))}" gravcomp="1">'
    ]
    joint = link.joint
    if joint is not None and joint.axis is not None:
        lines.append(
            f'{pad}  <joint name="{joint.name}" pos="{_fmt(joint.pose1[:3, 3])}" axis="{_fmt(joint.axis)}"'
            f' range="{_fmt(joint.range)}"/>'
        )
    collisions = [m for m in link.meshes if m.collision]
    for mesh in link.meshes:
        if mesh.collision:
            mass = link.mass / len(collisions)
            lines.append(f'{pad}  <geom class="collision" mesh="{mesh.name}" mass="{mass:.9g}"/>')
        else:
            lines.append(f'{pad}  <geom class="visual" mesh="{mesh.name}" rgba="{_fmt(mesh.rgba)}"/>')
    if not collisions:
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="{link.mass:.9g}" diaginertia="1e-6 1e-6 1e-6"/>')
    if "_sensor" in link.name:
        lines.append(f'{pad}  <site name="{link.name}" size="0.001"/>')
    for child in range(len(links)):
        if links[child].parent == index:
            lines += _emit(links, child, depth + 1)
    lines.append(f"{pad}</body>")
    return lines


def build_hand_mjcf(force: bool = False) -> Path:
    """Write the hand MJCF (cached by input hash) and return its path."""
    import trimesh

    source_hash = _source_hash()
    if not force and HAND_XML.exists() and _STAMP.exists():
        if json.loads(_STAMP.read_text()).get("source_sha256") == source_hash:
            return HAND_XML

    links = load_dg5f_dex01()
    mesh_dir = OUTPUT_DIR / "meshes"
    shutil.rmtree(mesh_dir, ignore_errors=True)  # drop meshes from earlier builds
    mesh_dir.mkdir(parents=True)
    assets = []
    for mesh in (m for link in links for m in link.meshes):
        if not (mesh_dir / f"{mesh.name}.stl").exists():
            trimesh.Trimesh(mesh.points, mesh.faces, process=False).export(mesh_dir / f"{mesh.name}.stl")
            assets.append(f'    <mesh name="{mesh.name}" file="meshes/{mesh.name}.stl"/>')
    xml = "\n".join([
        '<mujoco model="dg5f_right_dex01">',
        '  <compiler angle="radian" meshdir="." inertiafromgeom="true"/>',
        "  <default>",
        '    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0"/></default>',
        # Hand geoms collide with objects (contype 1) but not with each other (contype 2 only).
        (
            '    <default class="collision"><geom type="mesh" contype="2" conaffinity="1" group="3" rgba="0.5 0.5 0.5'
            ' 0.3"/></default>'
        ),
        "  </default>",
        "  <asset>",
        *assets,
        "  </asset>",
        "  <worldbody>",
        *_emit(links, 0, 2),
        "  </worldbody>",
        "</mujoco>",
        "",
    ])
    HAND_XML.write_text(xml)
    _STAMP.write_text(json.dumps({"source_sha256": source_hash}) + "\n")
    return HAND_XML


if __name__ == "__main__":
    print(build_hand_mjcf(force=True))
