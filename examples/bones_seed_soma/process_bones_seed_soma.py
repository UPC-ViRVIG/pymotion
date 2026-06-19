"""Process BONES-SEED SOMA BVH motions with PyMotion and SOMA."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pymotion.rotations.quat_torch as quat
import torch
from pymotion.io.bvh import BVH
from soma import SOMALayer
from soma.io import save_vertex_animation_usd

BVH_VISUALIZATION_SCALE = 0.01
SOMA_FPS = 120.0
UNIFORM_COLOR = np.array([0.2, 0.8, 1.0], dtype=np.float32)
PROPORTIONAL_COLOR = np.array([1.0, 0.6, 0.2], dtype=np.float32)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a BONES-SEED BVH as SOMA-ready local rotation matrices and Hips translation."""
    bvh = BVH()
    bvh.load(str(bvh_path))

    local_quats, local_positions, _, _, _, _ = bvh.get_data()
    joint_names = list(map(str, bvh.data["names"]))
    soma_joint_indices = [idx for idx, name in enumerate(joint_names) if "Root" not in name]
    if "Hips" not in joint_names:
        raise ValueError(f"Expected BONES-SEED BVH to contain a Hips joint: {bvh_path}")
    hips_idx = joint_names.index("Hips")

    hips_trans = torch.from_numpy(local_positions[:, hips_idx, :]).to(device=device, dtype=dtype)
    hips_trans = hips_trans * translation_scale
    local_quats = local_quats[:, soma_joint_indices, :]

    local_quats = torch.from_numpy(local_quats).to(device=device, dtype=dtype)
    local_rot_mats = quat.to_matrix(local_quats)

    return local_rot_mats, hips_trans


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
) -> dict[str, Any]:
    """Load one BONES-SEED motion variant and run it through SOMA."""
    motion_path = bones_seed_root / motion_meta[motion_path_key]
    shape_path = bones_seed_root / motion_meta[shape_path_key]

    print(f"Loading motion: {motion_meta['move_name']} from {motion_path}")
    identity_params, scale_params = load_shape_params(shape_path, device=device)
    local_rot_mats, hips_trans = load_bones_seed_motion(
        motion_path,
        device=device,
        dtype=identity_params.dtype,
        translation_scale=translation_scale,
    )

    print(f"local_rot_mats: {tuple(local_rot_mats.shape)}")
    print(f"hips_trans: {tuple(hips_trans.shape)}")

    soma_output = soma(
        local_rot_mats,
        identity_params,
        scale_params=scale_params,
        transl=hips_trans,
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
        conn.set_fps(SOMA_FPS)
        conn.render_checkerboard_floor()
        conn.render_bvh(
            uniform_bvh,
            color=UNIFORM_COLOR,
            exclude_end_sites=True,
            bvh_scale=BVH_VISUALIZATION_SCALE,
        )
        conn.render_bvh(
            proportion_bvh,
            color=PROPORTIONAL_COLOR,
            exclude_end_sites=True,
            bvh_scale=BVH_VISUALIZATION_SCALE,
        )


def save_soma_mesh_animation_to_temp_usd(
    soma: SOMALayer,
    soma_output: dict[str, Any],
    *,
    prefix: str,
) -> Path:
    """Save SOMA vertices as a temporary animated USD mesh."""
    vertices = soma_output["vertices"].detach().cpu().numpy()
    faces = soma.faces.detach().cpu().numpy()

    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".usdc", delete=False) as tmp_file:
        usd_path = Path(tmp_file.name)

    save_vertex_animation_usd(
        usd_path,
        vertices,
        faces,
        unit="meters",
        fps=SOMA_FPS,
    )
    return usd_path


def visualize_soma_meshes(
    soma: SOMALayer,
    uniform_output: dict[str, Any],
    proportion_output: dict[str, Any],
    blender_executable: Optional[Path],
    *,
    clear_scene: bool,
) -> None:
    """Render the selected SOMA mesh animations in Blender via temporary USD files."""
    from pymotion.render.blender import BlenderConnection

    uniform_usd_path = save_soma_mesh_animation_to_temp_usd(
        soma,
        uniform_output,
        prefix="pymotion_uniform_soma_",
    )
    proportion_usd_path = save_soma_mesh_animation_to_temp_usd(
        soma,
        proportion_output,
        prefix="pymotion_proportional_soma_",
    )

    print("\nVisualizing SOMA meshes in Blender")

    blender_executable_path = str(blender_executable) if blender_executable else None
    with BlenderConnection(blender_executable_path=blender_executable_path) as conn:
        if clear_scene:
            conn.clear_scene()
            conn.setup_rendering()
            conn.set_fps(SOMA_FPS)
            conn.render_checkerboard_floor()
        else:
            conn.set_fps(SOMA_FPS)
        conn.render_usd_from_path(str(uniform_usd_path), color=UNIFORM_COLOR, delete_after=True)
        conn.render_usd_from_path(
            str(proportion_usd_path),
            color=PROPORTIONAL_COLOR,
            delete_after=True,
        )


def select_motion(
    metadata: pd.DataFrame,
    motion_index: Optional[int],
    motion_name: Optional[str],
) -> tuple[int, pd.Series]:
    """Select a metadata row by index, exact motion name, or random draw."""
    if motion_name is not None:
        move_names = metadata["move_name"].astype(str)
        matches = np.flatnonzero(move_names.to_numpy() == motion_name)
        if len(matches) == 0:
            partial_matches = move_names[
                move_names.str.contains(motion_name, case=False, regex=False, na=False)
            ].head(5)
            suggestions = "\n".join(f"  - {name}" for name in partial_matches)
            message = f"No metadata row found with move_name={motion_name!r}."
            if suggestions:
                message += f"\nPartial matches:\n{suggestions}"
            raise ValueError(message)
        if len(matches) > 1:
            print(f"Found {len(matches)} rows named {motion_name!r}; using the first one.")
        motion_index = int(matches[0])
    elif motion_index is None:
        motion_index = int(np.random.randint(0, len(metadata)))

    if motion_index < 0 or motion_index >= len(metadata):
        raise ValueError(f"Motion index {motion_index} is out of range [0, {len(metadata) - 1}]")
    return motion_index, metadata.iloc[motion_index]


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
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--motion-index",
        type=int,
        default=None,
        help="Metadata row to process. Defaults to a random row.",
    )
    motion_group.add_argument(
        "--motion-name",
        default=None,
        help=(
            "Exact BONES-SEED move_name to process. Motion names can be copied "
            "from https://seed-viewer.bones.studio/."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.01,
        help="Scale applied to BVH Hips translations. BONES-SEED BVH files are typically in centimeters.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Render the selected uniform and proportional BVH skeletons in Blender.",
    )
    parser.add_argument(
        "--viz-mesh",
        action="store_true",
        help="Render the selected uniform and proportional SOMA meshes in Blender.",
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

    motion_idx, motion_meta = select_motion(metadata, args.motion_index, args.motion_name)
    print(f"Selected metadata row (motion): {motion_idx}")
    print(f"Selected motion name: {motion_meta['move_name']}")

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
    uniform_output = run_soma_motion(
        soma,
        motion_meta,
        bones_seed_root,
        motion_path_key="move_soma_uniform_path",
        shape_path_key="move_soma_uniform_shape_path",
        device=device,
        translation_scale=args.translation_scale,
    )

    print("\nProportional SOMA motion")
    proportion_output = run_soma_motion(
        soma,
        motion_meta,
        bones_seed_root,
        motion_path_key="move_soma_proportional_path",
        shape_path_key="move_soma_proportional_shape_path",
        device=device,
        translation_scale=args.translation_scale,
    )

    if args.viz_mesh:
        visualize_soma_meshes(
            soma,
            uniform_output,
            proportion_output,
            args.blender_executable,
            clear_scene=not args.viz,
        )


if __name__ == "__main__":
    main()
