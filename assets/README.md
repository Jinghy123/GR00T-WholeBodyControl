# Robot URDF assets

URDF-only copies (meshes stripped to keep the repo light). All files load with
`pinocchio.buildModelFromUrdf(path)` — kinematics-only, no mesh files needed.
Do NOT use `RobotWrapper.BuildFromURDF` on these (it builds geometry models and
requires the mesh files).

| dir | files | source | notes |
|---|---|---|---|
| `g1/` | `g1_body29_hand14.urdf`, `_virtual` variant | `psi/real/assets/g1/` | G1 29-DoF body + Dex3 14-DoF hands, fixed base (pelvis at origin). Used by `orchestration/unifolm/g1_eepose.py` for eepose FK. nq=43. |
| `unitree_hand/` | `unitree_dex3_{left,right}.urdf` | `psi/real/assets/unitree_hand/` | Standalone Dex3 hands, nq=7 each. Joint names identical to the hand joints inside `g1_body29_hand14.urdf`. |
| `inspire_hand/` | `inspire_hand_{left,right}.urdf` | `psi/real/assets/inspire_hand/` | Standalone Inspire five-finger hands, nq=12 each. |
| `brainco_hand/` | `revo2_{left,right}_hand.urdf`, `LICENSE` | [BrainCoTech/revo2_description](https://github.com/BrainCoTech/revo2_description) | BrainCo Revo2 five-finger hands (the BrainCo hand on G1), 6 active joints / nq=11 each. Apache-2.0. |

Meshes, if ever needed (visualization): G1/Dex3/Inspire meshes live in
`psi/real/assets/*/meshes/` (~100MB total); Revo2 meshes in the upstream repo.
