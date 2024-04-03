import numpy as np


def human_center_of_mass(
    joints_spine: np.array,
    joints_left_arm: np.array,
    joints_right_arm: np.array,
    joints_left_leg: np.array,
    joints_right_leg: np.array,
) -> np.array:
    """
    Compute the center of mass of a human body definedy by the standard human weight distribution.
    Each arm accounts for a 5% of the body weight, each leg accounts for a 15% of the body weight
    and the spine accounts for a 60% of the body weight.

    Parameters
    ----------
    joints_spine : np.array[..., n_joints, 3]
        Joint positions of the spine.
    joints_left_arm : np.array[..., n_joints, 3]
        Joint positions of the left arm.
    joints_right_arm : np.array[..., n_joints, 3]
        Joint positions of the right arm.
    joints_left_leg : np.array[..., n_joints, 3]
        Joint positions of the left leg.
    joints_right_leg : np.array[..., n_joints, 3]
        Joint positions of the right leg.

    Returns
    -------
    center_of_mass : np.array[..., 3]
        Center of mass.
    """
    n_joints_spine = joints_spine.shape[-2]
    n_joints_left_arm = joints_left_arm.shape[-2]
    n_joints_right_arm = joints_right_arm.shape[-2]
    n_joints_left_leg = joints_left_leg.shape[-2]
    n_joints_right_leg = joints_right_leg.shape[-2]
    weights = np.array(
        [0.6 / n_joints_spine] * n_joints_spine
        + [0.05 / n_joints_left_arm] * n_joints_left_arm
        + [0.05 / n_joints_right_arm] * n_joints_right_arm
        + [0.15 / n_joints_left_leg] * n_joints_left_leg
        + [0.15 / n_joints_right_leg] * n_joints_right_leg
    )
    joints = np.concatenate(
        [joints_spine, joints_left_arm, joints_right_arm, joints_left_leg, joints_right_leg], axis=-2
    )
    return center_of_mass(joints, weights)


def center_of_mass(joints: np.array, weights: np.array) -> np.array:
    """
    Compute the center of mass of a set of joints.

    Parameters
    ----------
    joints : np.array[..., n_joints, 3]
        Joint positions.
    weights : np.array[..., n_joints]
        Weights of the joints. The weights should sum to 1 along the last dimension.

    Returns
    -------
    center_of_mass : np.array[..., 3]
        Center of mass.
    """
    return np.sum(joints * weights[..., np.newaxis], axis=-2)
