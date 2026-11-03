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

"""Taxel layouts as NumPy arrays, matching ``dex01_sim_asset.taxel_patterns`` without Isaac Lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PATTERN_DIR = REPO_ROOT / "source/dex01_sim_asset/dex01_sim_asset/taxel_patterns"
DEX01_REFERENCE = json.loads((REPO_ROOT / "assets/sensors/dex01/reference.json").read_text())


@dataclass
class TaxelPattern:
    points: np.ndarray  # (N, 3) positions in the sensor frame
    axes: np.ndarray  # (N, 3, 3) rotations; column 2 is the outward surface normal
    taxel2pixel: np.ndarray  # (N, 2) row/column of each taxel in the tactile frame
    resolution: tuple[float, float]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return tuple(int(v) + 1 for v in self.taxel2pixel.max(0))


def grid_pattern(size: tuple[float, float], resolution: tuple[float, float]) -> TaxelPattern:
    """Regular grid from ``-size/2`` to ``size/2`` with +z normals ("xy" ordering)."""
    x = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution[0])
    y = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution[1])
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)
    rows, cols = np.meshgrid(np.arange(len(y)), np.arange(len(x)), indexing="ij")
    return TaxelPattern(
        points=points,
        axes=np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy(),
        taxel2pixel=np.stack([rows.ravel(), cols.ravel()], axis=1),
        resolution=resolution,
    )


def dex01_pattern() -> TaxelPattern:
    """DEX-01 fingertip: 740 contact samples on the overmold exterior, mapped into a 32 x 32 frame."""
    path = PATTERN_DIR / "dex01_contact.npz"
    with np.load(path, allow_pickle=False) as data:
        poses, taxel2pixel = data["poses"], data["taxel2pixel"]
    if poses.shape[1:] != (3, 4) or taxel2pixel.shape != (len(poses), 2):
        raise ValueError(f"Unexpected taxel pattern shapes in {path}; is it a git-lfs pointer?")
    return TaxelPattern(
        points=poses[:, :, 3].astype(float),
        axes=poses[:, :, :3].astype(float),
        taxel2pixel=taxel2pixel.astype(np.int64),
        resolution=(0.00072, 0.00088),
    )


def dex01_mount() -> tuple[np.ndarray, np.ndarray]:
    """Mount offset (position, wxyz quaternion) from the DG-5F sensor body to the pattern frame."""
    return (
        np.asarray(DEX01_REFERENCE["mount_translation_m"], dtype=float),
        np.asarray(DEX01_REFERENCE["mount_rotation_wxyz"], dtype=float),
    )
