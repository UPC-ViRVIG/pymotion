"""Process BONES-SEED SOMA BVH motions with PyMotion and SOMA."""

from __future__ import annotations

import argparse
import torch

import numpy as np
import pandas as pd
import pymotion.rotations.quat_torch as quat

from soma import SOMALayer
from pathlib import Path
from typing import Any, Optional
from pymotion.io.bvh import BVH

BVH_VISUALIZATION_SCALE = 0.01


def latest_metadata_path(bones_seed_root: Path) -> Path:
    """Return the newest seed_metadata*.csv file by sorted filename."""
    metadata_dir = bones_seed_root / "metadata"
    metadata_files = sorted(metadata_dir.glob("seed_metadata*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No seed_metadata*.csv files found in {metadata_dir}")
    return metadata_files[-1]


def load_bones_seed_motion(
    bvh_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    translation_scale: float,
    exclude_joint_substrings: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a BONES-SEED BVH as SOMA-ready local rotation matrices and root translation."""
    bvh = BVH()
    bvh.load(str(bvh_path))

    local_quats, local_positions, _, _, _, _ = bvh.get_data()
    root_trans = torch.from_numpy(local_positions[:, 0, :]).to(device=device, dtype=dtype)
    root_trans = root_trans * translation_scale

    if exclude_joint_substrings:
        keep_indices = [
            idx
            for idx, name in enumerate(bvh.data["names"])
            if not any(part in name for part in exclude_joint_substrings)
        ]
        if not keep_indices:
            raise ValueError(f"All joints were excluded from {bvh_path}")
        local_quats = local_quats[:, keep_indices, :]

    local_quats = torch.from_numpy(local_quats).to(device=device, dtype=dtype)
    local_rot_mats = quat.to_matrix(local_quats)

    return local_rot_mats, root_trans


def load_shape_params(path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SOMA identity and scale parameters from a BONES-SEED npz file."""
    shape_params = np.load(path)
    identity_params = torch.from_numpy(shape_params["identity_params"]).to(device)
    scale_params = torch.from_numpy(shape_params["scale_params"]).to(device)
    return identity_params, scale_params


def run_soma_motion(
    soma: SOMALayer,
    motion_meta: pd.Series,
    bones_seed_root: Path,
    *,
    motion_path_key: str,
    shape_path_key: str,
    device: torch.device,
    translation_scale: float,
    exclude_joint_substrings: tuple[str, ...],
) -> dict[str, Any]:
    """Load one BONES-SEED motion variant and run it through SOMA."""
    motion_path = bones_seed_root / motion_meta[motion_path_key]
    shape_path = bones_seed_root / motion_meta[shape_path_key]

    print(f"Loading motion: {motion_meta['move_name']} from {motion_path}")
    identity_params, scale_params = load_shape_params(shape_path, device=device)
    local_rot_mats, root_trans = load_bones_seed_motion(
        motion_path,
        device=device,
        dtype=identity_params.dtype,
        translation_scale=translation_scale,
        exclude_joint_substrings=exclude_joint_substrings,
    )

    print(f"local_rot_mats: {tuple(local_rot_mats.shape)}")
    print(f"root_trans: {tuple(root_trans.shape)}")

    soma_output = soma(
        local_rot_mats,
        identity_params,
        scale_params=scale_params,
        transl=root_trans,
        pose2rot=False,
        absolute_pose=True,
    )

    print(f"vertices: {tuple(soma_output['vertices'].shape)}")
    print(f"joints: {tuple(soma_output['joints'].shape)}")
    return soma_output


def visualize_bvh_skeletons(
    motion_meta: pd.Series,
    bones_seed_root: Path,
    blender_executable: Optional[Path],
) -> None:
    """Render the selected BONES-SEED BVH skeletons in Blender."""
    from pymotion.render.blender import BlenderConnection

    uniform_motion_path = bones_seed_root / motion_meta["move_soma_uniform_path"]
    proportion_motion_path = bones_seed_root / motion_meta["move_soma_proportional_path"]

    print("\nVisualizing BVH skeletons in Blender")
    print(f"Uniform BVH: {uniform_motion_path}")
    print(f"Proportional BVH: {proportion_motion_path}")
    print(f"Blender BVH import scale: {BVH_VISUALIZATION_SCALE}")

    uniform_bvh = BVH()
    uniform_bvh.load(str(uniform_motion_path))
    proportion_bvh = BVH()
    proportion_bvh.load(str(proportion_motion_path))

    blender_executable_path = str(blender_executable) if blender_executable else None
    with BlenderConnection(blender_executable_path=blender_executable_path) as conn:
        conn.clear_scene()
        conn.setup_rendering()
        conn.render_checkerboard_floor()
        conn.render_bvh(
            uniform_bvh,
            color=np.array([0.2, 0.8, 1.0], dtype=np.float32),
            exclude_end_sites=True,
            bvh_scale=BVH_VISUALIZATION_SCALE,
        )
        conn.render_bvh(
            proportion_bvh,
            color=np.array([1.0, 0.6, 0.2], dtype=np.float32),
            exclude_end_sites=True,
            bvh_scale=BVH_VISUALIZATION_SCALE,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load BONES-SEED SOMA BVH motions with PyMotion and run SOMA."
    )
    parser.add_argument(
        "--bones-seed-root",
        type=Path,
        required=True,
        help="Path to the BONES-SEED base directory.",
    )
    parser.add_argument(
        "--soma-assets",
        type=Path,
        required=True,
        help="Path to the SOMA assets directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for SOMA.",
    )
    parser.add_argument(
        "--motion-index",
        type=int,
        default=None,
        help="Metadata row to process. Defaults to a random row.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.01,
        help="Scale applied to BVH root translations. BONES-SEED BVH files are typically in centimeters.",
    )
    parser.add_argument(
        "--exclude-joint-substring",
        action="append",
        default=None,
        help=(
            "Exclude BVH joints whose names contain this substring before passing "
            "rotations to SOMA. Repeat for multiple substrings. Defaults to Root "
            "to match the BONES-SEED viewer parser."
        ),
    )
    parser.add_argument(
        "--keep-all-joints",
        action="store_true",
        help="Disable the default BONES-SEED Root joint filtering.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Render the selected uniform and proportional BVH skeletons in Blender.",
    )
    parser.add_argument(
        "--blender-executable",
        type=Path,
        default=None,
        help="Path to blender.exe. Use this if Blender is not installed in a default location or on PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bones_seed_root = args.bones_seed_root.expanduser().resolve()
    soma_assets = args.soma_assets.expanduser().resolve()
    device = torch.device(args.device)

    metadata_path = latest_metadata_path(bones_seed_root)
    metadata = pd.read_csv(metadata_path)
    print(f"Metadata: {metadata_path}")
    print(f"Number of motions: {len(metadata)}")

    motion_idx = args.motion_index
    if motion_idx is None:
        motion_idx = int(np.random.randint(0, len(metadata)))
    motion_meta = metadata.iloc[motion_idx]
    print(f"Selected metadata row (motion): {motion_idx}")

    exclude_joint_substrings = ()
    if not args.keep_all_joints:
        exclude_joint_substrings = tuple(args.exclude_joint_substring or ["Root"])

    if args.viz:
        visualize_bvh_skeletons(
            motion_meta,
            bones_seed_root,
            args.blender_executable,
        )

    soma = SOMALayer(
        data_root=str(soma_assets),
        identity_model_type="mhr",
        device=str(device),
    )

    print("\nUniform SOMA motion")
    run_soma_motion(
        soma,
        motion_meta,
        bones_seed_root,
        motion_path_key="move_soma_uniform_path",
        shape_path_key="move_soma_uniform_shape_path",
        device=device,
        translation_scale=args.translation_scale,
        exclude_joint_substrings=exclude_joint_substrings,
    )

    print("\nProportional SOMA motion")
    run_soma_motion(
        soma,
        motion_meta,
        bones_seed_root,
        motion_path_key="move_soma_proportional_path",
        shape_path_key="move_soma_proportional_shape_path",
        device=device,
        translation_scale=args.translation_scale,
        exclude_joint_substrings=exclude_joint_substrings,
    )


if __name__ == "__main__":
    main()
