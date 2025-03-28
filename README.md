# PyMotion: A Python Library for Motion Data

PyMotion is a Python library that provides various functions for manipulating and processing motion data in NumPy or PyTorch. It is designed to facilitate the development of neural networks for character animation.

Some features of PyMotion are:

- A comprehensive set of quaternion operations and conversions to other rotation representations, such as rotation matrix, axis-angle, euler, and 6D representation
- A dual quaternion representation for rigid displacements, which can help neural networks better understand poses, as proposed by [Andreou et al. [2022]](https://doi.org/10.1111/cgf.14632) and later adopted by [Ponton et al. [2023]](https://upc-virvig.github.io/SparsePoser/)
- A continuous 6D rotation representation, as introduced by [Zhou et al. [2019]](https://doi.org/10.1109/CVPR.2019.00589)
- A BVH file reader and preprocessor for loading and transforming motion data
- Skeletal operations such as Forward Kinematics for computing global joint positions from local joint rotations
- A plotly-based visualizer for debugging and visualizing character animation directly in Python
- [**Experimental**] PyMotion to Blender automatic communication for debugging and visualizing character animation
- NumPy and PyTorch implementations and tests for all functions

## Contents

1. [Installation](#installation)
2. [Examples](#examples)
3. [Roadmap](#roadmap)
4. [License](#license)

## Installation
1. **[Optional]** Install PyTorch using Pip as instructed in their [webpage](https://pytorch.org/get-started/locally/).

2. Install PyMotion:
```bash
pip install upc-pymotion
```

3. **[Optional]** Install Plotly and Dash for the visualizer (no needed for Blender visualization):
```bash
pip install upc-pymotion[viewer]
```

## Examples

<details>
<summary> Read and save a BVH file </summary>

```python
import numpy as np
from pymotion.io.bvh import BVH

bvh = BVH()
bvh.load("test.bvh")

print(bvh.data["names"])
# Example Output: ['Hips', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe', 'Chest', 'Chest3', 'Chest4', 'Neck', 'Head', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist']


# Move root joint to (0, 0, 0)
local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
local_positions[:, 0, :] = np.zeros((local_positions.shape[0], 3))
bvh.set_data(local_rotations, local_positions)

# Scale the skeleton
bvh.set_scale(0.75)

bvh.save("test_out.bvh")
```

</details>

<details>
<summary> Compute world positions and rotations from a BVH file </summary> <br/>

**NumPy**
```python
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk

bvh = BVH()
bvh.load("test.bvh")

local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
global_positions = local_positions[:, 0, :]  # root joint
pos, rotmats = fk(local_rotations, global_positions, offsets, parents)
```

**PyTorch**
```python
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton_torch import fk
import torch

bvh = BVH()
bvh.load("test.bvh")

local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
global_positions = local_positions[:, 0, :]  # root joint
pos, rotmats = fk(
    torch.from_numpy(local_rotations),
    torch.from_numpy(global_positions),
    torch.from_numpy(offsets),
    torch.from_numpy(parents),
)
```

</details>

<details>
<summary> Quaternion conversion to other representations </summary> <br/>

**NumPy**
```python
import pymotion.rotations.quat as quat
import numpy as np

angles = np.array([np.pi / 2, np.pi, np.pi / 4])[..., np.newaxis]
# angles.shape = [3, 1]
axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# axes.shape = [3, 3]

q = quat.from_angle_axis(angles, axes)

rotmats = quat.to_matrix(q)

euler = quat.to_euler(q, np.array([["x", "y", "z"], ["z", "y", "x"], ["y", "z", "x"]]))
euler_degrees = np.degrees(euler)

scaled_axis = quat.to_scaled_angle_axis(q)
```

**PyTorch**
```python
import pymotion.rotations.quat_torch as quat
import numpy as np
import torch

angles = torch.Tensor([torch.pi / 2, torch.pi, torch.pi / 4]).unsqueeze(-1)
# angles.shape = [3, 1]
axes = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# axes.shape = [3, 3]

q = quat.from_angle_axis(angles, axes)

rotmats = quat.to_matrix(q)

euler = quat.to_euler(q, np.array([["x", "y", "z"], ["z", "y", "x"], ["y", "z", "x"]]))
euler_degrees = torch.rad2deg(euler)

scaled_axis = quat.to_scaled_angle_axis(q)
```

</details>

<details>
<summary> Root-centered dual quaternions from a BVH file </summary> <br/>

**NumPy**
```python
from pymotion.io.bvh import BVH
import pymotion.ops.skeleton as sk
import numpy as np

bvh = BVH()
bvh.load("test.bvh")

local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

root_dual_quats = sk.to_root_dual_quat(
    local_rotations, local_positions[:, 0, :], parents, offsets
)

local_translations, local_rotations = sk.from_root_dual_quat(root_dual_quats, parents)
global_positions = local_translations[:, 0, :]
offsets = local_translations.copy()
offsets[:, 0, :] = np.zeros((offsets.shape[0], 3))
```

**PyTorch**
```python
from pymotion.io.bvh import BVH
import pymotion.ops.skeleton_torch as sk
import torch

bvh = BVH()
bvh.load("test.bvh")

local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

root_dual_quats = sk.to_root_dual_quat(
    torch.from_numpy(local_rotations),
    torch.from_numpy(local_positions[:, 0, :]),
    torch.from_numpy(parents),
    torch.from_numpy(offsets),
)

local_translations, local_rotations = sk.from_root_dual_quat(root_dual_quats, parents)
global_positions = local_translations[:, 0, :]
offsets = local_translations.clone()
offsets[:, 0, :] = torch.zeros((offsets.shape[0], 3))
```

</details>

<details>
<summary> 6D representation from a BVH file </summary> <br/>

**NumPy**
```python
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd

bvh = BVH()
bvh.load("test.bvh")

local_rotations, _, _, _, _, _ = bvh.get_data()

continuous = sixd.from_quat(local_rotations)

local_rotations = sixd.to_quat(continuous)
```

**PyTorch**
```python
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d_torch as sixd
import torch

bvh = BVH()
bvh.load("test.bvh")

local_rotations, _, _, _, _, _ = bvh.get_data()

continuous = sixd.from_quat(torch.from_numpy(local_rotations))

local_rotations = sixd.to_quat(continuous)
```

</details>

<details>
<summary> Skeleton local rotations from root-centered positions </summary> <br/>

**NumPy**
```python
import numpy as np
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from pymotion.ops.skeleton import from_root_positions

bvh = BVH()
bvh.load("test.bvh")
local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
pos, _ = fk(local_rotations, np.zeros((local_positions.shape[0], 3)), offsets, parents)

pred_rots = from_root_positions(pos, parents, offsets)

bvh.set_data(pred_rots, local_positions)
bvh.save("test_out.bvh")  # joint positions should be similar as test.bvh
```

**PyTorch**
```python
import torch
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton_torch import fk
from pymotion.ops.skeleton_torch import from_root_positions

bvh = BVH()
bvh.load("test.bvh")
local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
offsets = torch.from_numpy(offsets)
parents = torch.from_numpy(parents)
pos, _ = fk(
    torch.from_numpy(local_rotations),
    torch.zeros((local_positions.shape[0], 3)),
    offsets,
    parents,
)

pred_rots = from_root_positions(pos, parents, offsets)

bvh.set_data(pred_rots.numpy(), local_positions)
bvh.save("test_out.bvh")  # joint positions should be similar as test.bvh
```

</details>

<details>
<summary> Mirroring an animation </summary> <br/>

**NumPy**
```python
import numpy as np
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import mirror

bvh = BVH()
bvh.load("test.bvh")

names = bvh.data["names"].tolist()
print(names)
# ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 'RightCollar', 'RightShoulder',
#  'RightElbow', 'RightWrist', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightHip',
#  'RightKnee', 'RightAnkle', 'RightToe', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe']

joints_mapping = np.array(
    [
        (
            names.index("Left" + n[5:])
            if n.startswith("Right")
            else (names.index("Right" + n[4:]) if n.startswith("Left") else names.index(n))
        )
        for n in names
    ]
)

local_rotations, local_positions, parents, offsets, end_sites, _ = bvh.get_data()

mirrored_local_rots, mirrored_global_pos, mirrored_offsets, _ = mirror(
    local_rotations,
    local_positions[:, 0, :],  # global position of the root joint
    parents,
    offsets,
    end_sites,
    joints_mapping=joints_mapping,  # joints_mapping is only required for mode="symmetry"
    mode="symmetry",  # other modes: "all" or "positions"
    axis="X",
)

local_positions[:, 0, :] = mirrored_global_pos
bvh.set_data(mirrored_local_rots, local_positions)
# Uncomment when mode == "all"
# bvh.data["offsets"] = mirrored_offsets
bvh.save("test_mirrored.bvh")
```

**PyTorch**
```python
import torch
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton_torch import mirror

bvh = BVH()
bvh.load("test.bvh")

names = bvh.data["names"].tolist()
print(names)
# ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 'RightCollar', 'RightShoulder',
#  'RightElbow', 'RightWrist', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightHip',
#  'RightKnee', 'RightAnkle', 'RightToe', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe']

joints_mapping = torch.Tensor(
    [
        (
            names.index("Left" + n[5:])
            if n.startswith("Right")
            else (names.index("Right" + n[4:]) if n.startswith("Left") else names.index(n))
        )
        for n in names
    ]
).to(torch.int32)

local_rotations, local_positions, parents, offsets, end_sites, _ = bvh.get_data()

mirrored_local_rots, mirrored_global_pos, mirrored_offsets, _ = mirror(
    torch.from_numpy(local_rotations),
    torch.from_numpy(local_positions[:, 0, :]),  # global position of the root joint
    torch.from_numpy(parents),
    torch.from_numpy(offsets),
    torch.from_numpy(end_sites),
    joints_mapping=joints_mapping,  # joints_mapping is only required for mode="symmetry"
    mode="symmetry",  # other modes: "all" or "positions"
    axis="X",
)

local_positions[:, 0, :] = mirrored_global_pos.numpy()
bvh.set_data(mirrored_local_rots.numpy(), local_positions)
# Uncomment when mode == "all"
# bvh.data["offsets"] = mirrored_offsets.numpy()
bvh.save("test_mirrored.bvh")
```

</details>

<details>
<summary> Visualize motion in Python </summary> <br/>

```python
from pymotion.render.viewer import Viewer
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk

bvh = BVH()
bvh.load("test.bvh")

local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
global_positions = local_positions[:, 0, :]  # root joint
pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

viewer = Viewer(use_reloader=True, xy_size=5)
viewer.add_skeleton(pos, parents)
# add additional info using add_sphere(...) and/or add_line(...), examples:
# viewer.add_sphere(sphere_pos, color="green")
# viewer.add_line(start_pos, end_pos, color="green")
viewer.add_floor()
viewer.run()
```

</details>

<details>
<summary> Visualize points, orientations and BVH files in Blender </summary> <br/>

```python
import numpy as np
from pymotion.io.bvh import BVH
from pymotion.render.blender import BlenderConnection

with BlenderConnection() as conn:
    conn.clear_scene()
    conn.render_checkerboard_floor()
    conn.render_points(
        np.array([[0, -3, 0], [1, 2, 3]]), np.array([[0, 0, 1], [0, 1, 0]]), radius=np.array([[0.25], [0.05]])
    )
    conn.render_orientations(
        np.array([[1, 0, 0, 0], [np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0, 0]]),
        np.array([[0, -3, 0], [1, 2, 3]]),
        scale=np.array([[0.5], [0.25]]),
    )
    # BVH files can be rendered directly from file path
    path = "test.bvh"
    conn.render_bvh_from_path(
        path,
        np.array([0, 0, 1]),
        end_joints=["RightWrist", "LeftWrist", "RightToe", "LeftToe", "Head"],
    )
    # or by using a BVH object
    bvh = BVH()
    path = "test2.bvh"
    bvh.load(path)
    conn.render_bvh(
        bvh, np.array([0, 1, 0]), end_joints=["RightWrist", "LeftWrist", "RightToe", "LeftToe", "Head"]
    )
```

</details>

## Roadmap

This repository is authored and maintained by [Jose Luis Ponton](https://github.com/JLPM22) as part of his Ph.D.

Features will be added when new operations or rotation representations are needed in the development of research projects. Here it is a list of possible features and improvements for the future:

- Extend documentation and add examples in the description of each function.
- Include new animation importers such as FBX
- Include useful operations for data augmentation such as *animation mirroring (done)*, noise addition, temporal warping, etc.
- Create an Inverse Kinematics module

## License

This work is licensed under the MIT license. Please, see the [LICENSE](LICENSE) for further details.
