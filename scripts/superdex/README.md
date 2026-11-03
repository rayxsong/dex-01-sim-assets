# SuperDex examples

[SuperDex Physics](https://github.com/facebookresearch/project_superdex) ports of the Isaac Lab
examples. No Isaac Sim, Isaac Lab, or NVIDIA GPU is required. SuperDex needs Python 3.12.

## Setup

```bash
git lfs pull
uv venv --python 3.12 ../env_dex01_superdex
source ../env_dex01_superdex/bin/activate
uv pip install -e source/dex01_common -e source/dex01_superdex
```

The DG-5F + DEX-01 hand is read from `assets/robots/tesollo/dg5f_right.usda` (fingertip variant
`dex01`) by `dex01_common.usd_hand` and built as a SuperDex articulation at startup.

## Examples

| Script | Scene | Expected output |
|---|---|---|
| `simple_example.py` | 0.5 kg gear dropped on a flat 32 x 32 pad | Settled total 4.905 N |
| `dex01_on_tesollo_dg5f.py` | 0.1 kg gear rests on fingers 2-5 in turn (`--flat-indenter` for a 6 mm cube) | Loaded finger 0.981 N, others 0 |
| `dex01_stationary_press.py` | Finger 2 articulates against a fixed gear | Hold load near `--press-force-n` (1 N), 0 after release |

Each opens a [Rerun](https://rerun.io) viewer with the scene, taxels coloured by force, the
firmware frame of every sensor and a plot of each sensor's total normal force. Add
`--save run.rrd` to record a run, or `--headless --max-steps N` for a non-interactive run.

## Differences from the Isaac Lab model

- The sensor model is the same: the counter object's SDF gives taxel penetration depths, and the
  measured net contact force is resolved along the depth-weighted normal and redistributed by depth.
  SuperDex provides both the filtered actor-to-actor contact force and the SDF query, so
  non-convex counter objects such as the gear's teeth are exact.
- Scenes are Z-up with gravity along -Z (SuperDex Physics defaults to Y-up).
- Contact uses a zero penalty threshold with 0.05 mm smoothing on the tactile actors. SuperDex's
  default penalty starts 1 mm before touching, which leaves light objects resting in a gap where no
  taxel penetrates.
- The DEX-01 shell uses SuperDex's SDF collider, matching the USD's `sdf` approximation; other hand
  links use the convex hulls their USD collision approximations specify.
- A held object is a dynamic actor pinned by boundary conditions; a static actor has no contact
  sample points and does not collide with the SDF shell.
- NVIDIA's Factory gear is not redistributable; the gear here is an extruded spur-gear outline.
- The stationary press integrates half of each joint correction into its previous targets, with a
  0.002 rad per-step limit; stepping straight to the full correction chatters against SuperDex's
  stiffer penalty contact.
