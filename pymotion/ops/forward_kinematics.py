import numpy as np
import pymotion.rotations.quat as quat


def fk(
    rot: np.array,
    global_pos: np.array,
    offsets: np.array,
    parents: np.array,
) -> np.array:
    """
    Parameters
    -----------
        rot: np.array[..., n_joints, 4]
        global_pos: np.array[..., 3]
        offsets: np.array[..., n_joints, 3] or np.array[n_joints, 3]
        parents: np.array[n_joints]

    Returns
    --------
        positions: np.array[..., n_joints, 3]
            positions of the joints
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.
            rotation matrices of the joints

    Example
    --------
        >>> from pymotion.io.bvh import BVH
        >>> bvh = BVH()
        >>> bvh.load("test.bvh")

        >>> import pymotion.rotations.quat as quat
        >>> qs = quat.from_euler(np.radians(bvh.data["rotations"]), order=bvh.data["rot_order"])

        >>> pos, rotmats = fk(qs, bvh.data["positions"][:, 0, :], bvh.data["offsets"], bvh.data["parents"])
    """
    # create a homogeneous matrix of shape (..., 4, 4)
    mat = np.zeros(rot.shape[:-1] + (4, 4))
    mat[..., :3, :3] = quat.to_matrix(quat.normalize(rot))
    mat[..., :3, 3] = offsets
    mat[..., 3, 3] = 1
    # first joint is global position
    mat[..., 0, :3, 3] = global_pos
    # other joints are transformed by the transform matrix
    for i, parent in enumerate(parents):
        # root
        if i == 0:
            continue
        mat[..., i, :, :] = np.matmul(
            mat[..., parent, :, :],
            mat[..., i, :, :],
        )
    positions = mat[..., :3, 3]
    rotmats = mat[..., :3, :3]
    return positions, rotmats
