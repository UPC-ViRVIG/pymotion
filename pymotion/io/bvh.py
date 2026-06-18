# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os

import numpy as np
import pymotion.rotations.quat_np as quat
import pymotion.ops.vector as vec
import pymotion.ops.skeleton as skeleton

from pymotion.ops.time import savgol_filter


class BVH:
    def __init__(self):
        self.bvh_rot_map = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}
        self.bvh_map_num = {"x": 0, "y": 1, "z": 2}
        self.bvh_pos_map_num = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
        self.inv_bvh_rot_map = {v: k for k, v in self.bvh_rot_map.items()}

    def copy(self, bvh):
        """
        Deep copies a BVH object.

        Parameters
        ----------
        bvh : BVH
            BVH object to copy.
        """
        self.data = copy.deepcopy(bvh.data)

    def load(self, filepath: str):
        """
        Reads a BVH file and returns a dictionary with the data.

        Parameters
        ----------
        filepath : str
            path to the file.

        Results
        ------
        self.data : dict
            dictionary with the data.
            ["filename"] : str
                name of the file.
            ["names"] : np.array[str]
                ith-element contain the name of ith-joint.
            ["offsets"] : np.array[n_joints, 3]
                ith-element contain the offset of ith-joint wrt. its parent joint.
            ["end_sites"] : np.array[n_end_sites, 3]
                ith-element contain the offset of ith-end-site wrt. its parent joint.
            ["end_sites_parents"] : np.array[int]
                ith-element contain the joint parent of the ith end-site.
            ["parents"] : np.array[int]
                ith-element contain the parent of the ith joint. The root joint has parent 0 (itself).
            ["rot_order"] : np.array[n_joints, 3]
                order per channel of the rotations. The order is 'x', 'y' or 'z'.
            ["positions"] : np.array[n_frames, n_joints, 3]
                local positions.
            ["rotations"] : np.array[n_frames, n_joints, 3]
                local rotations in euler angles with the order specified in rot_order.
            ["frame_time"] : float
                time between two frames in seconds.
            ["channels"] : np.array[int]
                ith-element contain the number of channels of the ith joint.
        """
        f = open(filepath, "r")
        filename = os.path.basename(filepath)
        filename = filename.split(".")[:-1]
        filename = ".".join(filename)

        names = []
        offsets = []
        end_sites = []
        end_sites_parents = []
        parents = []
        position_order = []
        rot_order = []
        channels = []

        current = None
        is_end_site = False
        reading_frames = False
        frame = 0

        for line in f:
            if not reading_frames:
                if "HIERARCHY" in line or "MOTION" in line or "{" in line:
                    continue

                if "ROOT" in line or "JOINT" in line:
                    names.append(line.split()[1])
                    offsets.append(None)
                    parents.append(current)
                    position_order.append(None)
                    rot_order.append(None)
                    channels.append(None)
                    current = len(names) - 1
                    continue

                if "}" in line:
                    if is_end_site:
                        is_end_site = False
                    else:
                        current = parents[current]
                    continue

                if "End Site" in line:
                    is_end_site = True
                    end_sites_parents.append(current)
                    end_sites.append(None)
                    continue

                if "OFFSET" in line:
                    if is_end_site:
                        end_sites[-1] = [float(x) for x in line.split()[1:4]]
                    else:
                        offsets[current] = [float(x) for x in line.split()[1:4]]
                    continue

                if "CHANNELS" in line:
                    words = line.split()
                    number_channels = int(words[1])
                    channels[current] = number_channels
                    if number_channels == 6:
                        position_order[current] = [self.bvh_pos_map_num[x] for x in words[2 : 2 + 3]]
                        rot_order[current] = [self.bvh_rot_map[x] for x in words[2 + 3 : 2 + 3 + 3]]
                    elif number_channels == 3:
                        rot_order[current] = [self.bvh_rot_map[x] for x in words[2 : 2 + 3]]
                    else:
                        raise Exception("Unknown number of channels")
                    continue

                if "Frames" in line:
                    names = np.array(names)
                    number_frames = int(line.split()[1])
                    offsets = np.array(offsets)
                    parents[0] = 0
                    parents = np.array(parents)
                    end_sites = np.array(end_sites)
                    end_sites_parents = np.array(end_sites_parents)
                    rot_order = np.array(rot_order)
                    channels = np.array(channels)
                    positions = np.tile(offsets, (number_frames, 1)).reshape(number_frames, len(offsets), 3)
                    rotations = np.zeros((number_frames, len(names), 3))
                    continue

                if "Frame Time" in line:
                    frame_time = float(line.split()[2])
                    reading_frames = True
                    continue
            else:
                values = [float(x) for x in line.split()]
                i = 0
                for j in range(len(names)):
                    if channels[j] == 6:
                        positions[frame, j, position_order[j]] = values[i : i + 3]
                        rotations[frame, j] = values[i + 3 : i + 6]
                    elif channels[j] == 3:
                        rotations[frame, j] = values[i : i + 3]
                    i += channels[j]
                frame += 1

        f.close()

        self.data = {
            "filename": filename,
            "names": names,
            "offsets": offsets,
            "end_sites": end_sites,
            "end_sites_parents": end_sites_parents,
            "parents": parents,
            "rot_order": rot_order,
            "channels": channels,  # TODO: properly handle different positional channels and orders
            "positions": positions,
            "rotations": rotations,
            "frame_time": frame_time,
        }

    def save(self, filename: str):
        """
        Saves a BVH file from a dictionary with the data.

        Parameters
        ----------
        filename : str
            path to the file.
        data : dict
            dictionary with the data following the
            returned dict structure from load(...).
            positions in data["positions"] are assumed to be X,Y,Z order.
            rotations in data["rotations"] are assumed to be in the specified order in data["rot_order"].
        """

        with open(filename, "w") as f:
            tab = ""
            f.write("%sHIERARCHY\n" % tab)
            f.write("%sROOT %s\n" % (tab, self.data["names"][0]))
            f.write("%s{\n" % tab)
            tab += "\t"

            f.write(
                "%sOFFSET %f %f %f\n"
                % (
                    tab,
                    self.data["offsets"][0, 0],
                    self.data["offsets"][0, 1],
                    self.data["offsets"][0, 2],
                )
            )
            f.write(
                "%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n"
                % (
                    tab,
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 0]],
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 1]],
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 2]],
                )
            )

            joint_order = [0]

            for i in range(len(self.data["parents"])):
                if self.data["parents"][i] == 0 and i != 0:
                    tab = self._save_joint(f, self.data, tab, i, joint_order)

            tab = tab[:-1]
            f.write("%s}\n" % tab)

            f.write("%sMOTION\n" % tab)
            f.write("%sFrames: %d\n" % (tab, self.data["positions"].shape[0]))
            f.write("%sFrame Time: %f\n" % (tab, self.data["frame_time"]))

            for i in range(self.data["positions"].shape[0]):
                for j in joint_order:
                    if self.data["channels"][j] == 6:  # joint has position and rotation channels
                        f.write(
                            "%f %f %f "
                            % (
                                self.data["positions"][i, j, 0],
                                self.data["positions"][i, j, 1],
                                self.data["positions"][i, j, 2],
                            )
                        )
                    f.write("%f %f %f " % tuple(self.data["rotations"][i, j]))
                f.write("\n")

    def set_scale(self, scale: float):
        """
        Sets the scale of the BVH.

        Parameters
        ----------
        scale : float
            scale to apply to the BVH.
        """

        self.data["offsets"] *= scale
        self.data["end_sites"] *= scale
        self.data["positions"] *= scale

    def set_order_joints(self, order: list[int]):
        """
        Sets the order of the joints in the .bvh file.

        Parameters
        ----------
        order : list[int]
            for each joint j, order[j] is the new index of the joint j.

        """

        assert order[0] == 0, "root joint should not change"
        assert len(order) == len(
            self.data["names"]
        ), "order should have the same number of joints as the original bvh file"

        reverse_order = [order.index(i) for i in range(len(order))]

        self.data["names"] = np.array([self.data["names"][reverse_order[i]] for i in range(len(order))])
        self.data["offsets"] = self.data["offsets"][reverse_order]
        self.data["end_sites_parents"] = np.array([order[j] for j in self.data["end_sites_parents"]])
        self.data["parents"] = np.array(
            [order[self.data["parents"][reverse_order[i]]] for i in range(len(order))]
        )
        self.data["rot_order"] = self.data["rot_order"][reverse_order]
        self.data["positions"] = self.data["positions"][:, reverse_order]
        self.data["rotations"] = self.data["rotations"][:, reverse_order]

    def remove_joints(self, delete_joints: list[int]):
        """
        Removes joints from the .bvh file.

        Parameters
        ----------
        delete_joints : list[int]
            list of joint indices to remove.
        """

        # Identify joints to keep
        delete_joints_set = set(delete_joints)
        keep_joints = [i for i in range(len(self.data["names"])) if i not in delete_joints_set]
        keep_joints_set = set(keep_joints)
        new_to_old = dict(enumerate(keep_joints))
        old_to_new = dict((v, k) for k, v in new_to_old.items())

        # Update transforms for remaining joints
        rots, pos, parents, offsets, end_sites, end_sites_parents = self.get_data()
        new_rots = rots[:, keep_joints, :]
        new_pos = pos[:, keep_joints, :]
        new_offsets = offsets[keep_joints, :]
        for j in keep_joints:
            while parents[j] not in keep_joints_set:
                p = parents[j]
                new_rots[:, old_to_new[j], :] = quat.mul(rots[:, p, :], new_rots[:, old_to_new[j], :])
                new_pos[:, old_to_new[j], :] = (
                    quat.mul_vec(rots[:, p, :], new_pos[:, old_to_new[j], :]) + pos[:, p, :]
                )
                new_offsets[old_to_new[j]] = (
                    quat.mul_vec(rots[0, p, :], new_offsets[old_to_new[j]]) + offsets[p]
                )
                if parents[j] == 0:
                    # root
                    parents[j] = j
                    break
                else:
                    parents[j] = parents[p]

        # Update parent indices for remaining joints
        new_parents = [0] * len(keep_joints)
        for i, p in enumerate(parents):
            if i in keep_joints_set:
                new_parents[old_to_new[i]] = old_to_new[p]

        # Update end_sites_parents to reflect the removal of joints
        updated_end_sites_parents = [
            old_to_new[es_parent] for es_parent in end_sites_parents if es_parent in old_to_new
        ]
        # Remove end sites associated with deleted joints, if necessary
        valid_end_sites_indices = [
            i for i, es_parent in enumerate(end_sites_parents) if es_parent not in delete_joints
        ]

        # Update data
        self.data["names"] = np.array([self.data["names"][i] for i in keep_joints])
        self.data["rot_order"] = self.data["rot_order"][..., keep_joints, :]
        self.data["positions"] = new_pos
        self.data["rotations"] = np.degrees(
            quat.to_euler(new_rots, order=np.tile(self.data["rot_order"], (rots.shape[0], 1, 1)))
        )
        self.data["parents"] = np.array(new_parents)
        self.data["offsets"] = new_offsets
        self.data["end_sites"] = end_sites[valid_end_sites_indices]
        self.data["end_sites_parents"] = np.array(updated_end_sites_parents)

    def get_data(self):
        """
        Returns unrolled rotations (transformed to quaternions),
        positions, parents and offsets.

        Returns
        -------
        rots : np.array[n_frames, n_joints, 4]
            unrolled local rotations (transformed to quaternions).
        pos : np.array[n_frames, n_joints, 3]
            local positions.
        parents : np.array[int]
            ith-element contain the parent of the ith joint.
        offsets : np.array[n_joints, 3]
            ith-element contain the offset of ith-joint wrt. its parent joint.
        end_sites : np.array[n_joints, 3]
            ith-element contain the offset of ith-end-site wrt. its parent joint.
        end_sites_parents : np.array[int]
            ith-element contain the joint parent of the ith end-site.
        """
        rots = quat.unroll(
            quat.from_euler(
                np.radians(self.data["rotations"]),
                order=np.tile(self.data["rot_order"], (self.data["rotations"].shape[0], 1, 1)),
            ),
            axis=0,
        )
        rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
        pos = self.data["positions"]
        parents = self.data["parents"]
        offsets = self.data["offsets"]
        end_sites = self.data["end_sites"]
        end_sites_parents = self.data["end_sites_parents"]
        return rots, pos, parents, offsets, end_sites, end_sites_parents

    def set_data(self, rots, pos):
        """
        Sets the data of the BVH from rotations represented as quaternions,
        positions, parents and offsets.

        Parameters
        ----------
        rots : np.array[n_frames, n_joints, 4]
            local rotations (quaternions).
        pos : np.array[n_frames, n_joints, 3]
            local positions.
        """
        assert (
            self.data is not None
        ), "load a BVH file first or create a self.data dict with the same structure as the one returned by load(...)"
        assert (
            self.data["rot_order"] is not None
        ), "load a BVH file first or create a self.data dict with the same structure as the one returned by load(...)"

        self.data["rotations"] = np.degrees(
            quat.to_euler(rots, order=np.tile(self.data["rot_order"], (rots.shape[0], 1, 1)))
        )
        self.data["positions"] = pos

    def set_rotation_order(self, new_order: str):
        """
        Changes the rotation order of all joints in the BVH.
        Converts rotations from the current order to the new order.

        Parameters
        ----------
        new_order : str
            New rotation order, e.g., 'xyz', 'zyx', 'zxy', etc.
            Must be a 3-character string containing 'x', 'y', and 'z'.
        """
        new_order = new_order.lower()
        assert len(new_order) == 3, "Rotation order must be a 3-character string"
        assert set(new_order) == {
            "x",
            "y",
            "z",
        }, "Rotation order must contain 'x', 'y', and 'z'"

        # Get current data (converts euler angles to quaternions using current order)
        rots, pos, _, _, _, _ = self.get_data()

        # Update rotation order for all joints
        new_rot_order = np.array([[c for c in new_order] for _ in range(len(self.data["names"]))])
        self.data["rot_order"] = new_rot_order

        # Set data back (converts quaternions to euler angles using new order)
        self.set_data(rots, pos)

    def merge(self, other):
        """
        Merge another BVH into this one by appending its frames.
        Both BVHs must have the same skeleton (names, parents, rotation order).

        Parameters
        ----------
        other : BVH
            BVH object whose frames will be appended.
        """
        if np.any(self.data["names"] != other.data["names"]):
            raise ValueError("Names must match")
        if np.any(self.data["parents"] != other.data["parents"]):
            raise ValueError("Parents must match")
        if np.any(self.data["rot_order"] != other.data["rot_order"]):
            raise ValueError("Rotation orders must match")

        self.data["rotations"] = np.concatenate(
            [self.data["rotations"], other.data["rotations"]], axis=0
        )
        self.data["positions"] = np.concatenate(
            [self.data["positions"], other.data["positions"]], axis=0
        )

    def resample(self, target_frame_time: float = None, target_fps: float = None, source_fps: float = None):
        """
        Resample the animation to a different frame rate.
        Exactly one of target_frame_time or target_fps must be provided.

        Parameters
        ----------
        target_frame_time : float, optional
            Target time between frames in seconds.
        target_fps : float, optional
            Target frames per second.
        source_fps : float, optional
            If provided, overrides the frame_time stored in the BVH data.
        """
        if (target_frame_time is None) == (target_fps is None):
            raise ValueError("Exactly one of target_frame_time or target_fps must be provided.")
        if target_fps is not None:
            target_frame_time = 1.0 / target_fps

        if source_fps is not None:
            self.data["frame_time"] = 1.0 / source_fps

        frame_time = self.data["frame_time"]
        local_rotations, local_positions, _, _, _, _ = self.get_data()

        current_fps = int(round(1.0 / frame_time))
        target_fps_int = int(round(1.0 / target_frame_time))

        if current_fps // 2 == target_fps_int:
            # Exact 2x downsampling
            local_rotations = local_rotations[::2, ...]
            local_positions = local_positions[::2, ...]
            self.set_data(local_rotations, local_positions)
        elif current_fps == target_fps_int:
            # Same FPS, just update frame_time
            pass
        else:
            # General resampling via interpolation
            num_frames = local_rotations.shape[0]
            num_frames_target = int(round(num_frames * frame_time / target_frame_time))
            samples = np.linspace(0, num_frames - 1, num_frames_target)

            # Interpolate rotations (quaternions)
            rot_shape = local_rotations.shape
            reshaped_rots = local_rotations.reshape(rot_shape[0], -1)
            interpolated_rots = np.zeros((len(samples), reshaped_rots.shape[1]))
            for i in range(reshaped_rots.shape[1]):
                interpolated_rots[:, i] = np.interp(
                    x=samples,
                    xp=np.arange(num_frames),
                    fp=reshaped_rots[:, i],
                    left=reshaped_rots[-1, i],
                    right=reshaped_rots[0, i],
                )
            local_rotations = quat.normalize(
                interpolated_rots.reshape(len(samples), *rot_shape[1:])
            )

            # Interpolate positions
            pos_shape = local_positions.shape
            reshaped_pos = local_positions.reshape(pos_shape[0], -1)
            interpolated_pos = np.zeros((len(samples), reshaped_pos.shape[1]))
            for i in range(reshaped_pos.shape[1]):
                interpolated_pos[:, i] = np.interp(
                    x=samples,
                    xp=np.arange(num_frames),
                    fp=reshaped_pos[:, i],
                    left=reshaped_pos[-1, i],
                    right=reshaped_pos[0, i],
                )
            local_positions = interpolated_pos.reshape(len(samples), *pos_shape[1:])

            self.set_data(local_rotations, local_positions)

        self.data["frame_time"] = target_frame_time

    def keep_joints(self, joints: list):
        """
        Keep only the joints at the specified indices, removing the rest.

        Parameters
        ----------
        joints : list[int]
            List of joint indices to keep.
        """
        delete_joints = [
            i for i in range(len(self.data["names"])) if i not in joints
        ]
        self.remove_joints(delete_joints)

    def remove_joints_name(self, joint_names: list):
        """
        Remove joints specified by their names.

        Parameters
        ----------
        joint_names : list[str]
            List of joint names to remove.
        """
        joint_names_set = set(joint_names)
        indices = [
            i for i, name in enumerate(self.data["names"]) if name in joint_names_set
        ]
        self.remove_joints(indices)

    def keep_joints_name(self, joint_names: list):
        """
        Keep only joints with the given names, removing the rest.

        Parameters
        ----------
        joint_names : list[str]
            List of joint names to keep.
        """
        joint_names_set = set(joint_names)
        indices = [
            i for i, name in enumerate(self.data["names"]) if name not in joint_names_set
        ]
        self.remove_joints(indices)

    def remove_root_joint(self):
        """
        Remove the root joint (index 0) from the skeleton.
        """
        self.remove_joints([0])

    def slice_frames(self, start: int = 0, end: int = None):
        """
        Slice the animation frames using Python-style start/end indexing.

        Parameters
        ----------
        start : int, optional
            Start frame index (inclusive). Default is 0.
        end : int, optional
            End frame index (exclusive). Default is None (end of animation).
        """
        self.data["positions"] = self.data["positions"][start:end]
        self.data["rotations"] = self.data["rotations"][start:end]

    def joint_intersection(self, other):
        """
        Keep only joints whose names exist in both this BVH and the other BVH.

        Parameters
        ----------
        other : BVH
            Reference BVH for intersection.
        """
        other_names = set(other.data["names"])
        intersection = [name for name in self.data["names"] if name in other_names]
        self.keep_joints_name(intersection)

    def loop(self, num_loops: int):
        """
        Repeat the animation a given number of times.

        Parameters
        ----------
        num_loops : int
            Number of times to repeat the animation.
        """
        self.data["positions"] = np.concatenate(
            [self.data["positions"]] * num_loops, axis=0
        )
        self.data["rotations"] = np.concatenate(
            [self.data["rotations"]] * num_loops, axis=0
        )

    def shift(self, shift_frames: int):
        """
        Roll/shift the animation by moving the last N frames to the beginning.

        Parameters
        ----------
        shift_frames : int
            Number of frames to shift.
        """
        self.data["positions"] = np.concatenate(
            [
                self.data["positions"][-shift_frames:],
                self.data["positions"][:-shift_frames],
            ],
            axis=0,
        )
        self.data["rotations"] = np.concatenate(
            [
                self.data["rotations"][-shift_frames:],
                self.data["rotations"][:-shift_frames],
            ],
            axis=0,
        )

    def add_root_joint(self, local_forward_hips: np.ndarray):
        """
        Add a new root joint above the current root by extracting
        a smoothed character-space trajectory from the current root joint.

        Parameters
        ----------
        local_forward_hips : np.ndarray[3]
            The local forward direction of the hips joint,
            e.g., [0, 1, 0] or [0, 0, 1] depending on the capture system.
        """
        local_rotations, local_positions, _, _, _, _ = self.get_data()
        fps = int(round(1.0 / self.data["frame_time"]))

        global_rotations = local_rotations[:, 0, :]
        global_positions = local_positions[:, 0, :]

        # Smoothed character position (projection of pelvis to floor)
        window_length = fps + (1 - fps % 2)  # ensure odd
        character_positions = global_positions.copy()
        character_positions[:, 1] = 0.0
        character_positions = savgol_filter(
            character_positions, window_length, 3, axis=0, mode="nearest"
        )

        # Smoothed forward direction
        forward_hips = quat.mul_vec(global_rotations, local_forward_hips)
        forward_hips[:, 1] = 0.0
        forward_character = vec.normalize(forward_hips)
        forward_character = savgol_filter(
            forward_character, window_length, 3, axis=0, mode="nearest"
        )
        forward_character = vec.normalize(forward_character)

        # Character rotation from forward direction
        character_rotations = quat.from_to_axis(
            np.tile(np.array([0.0, 0.0, 1.0]), (len(global_rotations), 1)),
            forward_character,
            rot_axis=np.tile(np.array([0.0, 1.0, 0.0]), (len(global_rotations), 1)),
            normalize_input=False,
        )

        # Build new arrays with prepended root joint
        r_local_rotations = np.concatenate(
            [local_rotations[:, 0:1, :].copy(), local_rotations.copy()], axis=1
        )
        r_local_positions = np.concatenate(
            [local_positions[:, 0:1, :].copy(), local_positions.copy()], axis=1
        )

        # Joint 1 becomes relative to new root
        r_local_rotations[:, 1, :] = quat.mul(
            quat.inverse(character_rotations), global_rotations
        )
        r_local_positions[:, 1, :] = quat.mul_vec(
            quat.inverse(character_rotations),
            global_positions - character_positions,
        )

        # New root gets character-space transform
        r_local_rotations[:, 0, :] = character_rotations
        r_local_positions[:, 0, :] = character_positions

        # Update skeleton data
        self.data["names"] = np.concatenate(
            [np.array(["Root"]), self.data["names"]], axis=0
        )
        self.data["offsets"] = np.concatenate(
            [
                np.zeros((1, 3), dtype=np.float32),
                r_local_positions[0, 1:2].copy(),
                self.data["offsets"][1:],
            ],
            axis=0,
        )
        self.data["end_sites_parents"] = self.data["end_sites_parents"] + 1
        self.data["parents"] = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
                self.data["parents"][1:] + 1,
            ],
            axis=0,
        )
        self.data["rot_order"] = np.concatenate(
            [self.data["rot_order"][0:1], self.data["rot_order"]], axis=0
        )
        self.data["channels"] = np.concatenate(
            [np.array([6], dtype=np.int32), self.data["channels"]], axis=0
        )
        self.set_data(r_local_rotations, r_local_positions)

    def rotate_root(self, angle_degrees: float, axis: str = "y"):
        """
        Apply a global rotation around the specified axis to the root joint.
        Rotates both the root joint's rotation and translation.

        Parameters
        ----------
        angle_degrees : float
            Rotation angle in degrees.
        axis : str, optional
            Axis to rotate around: 'x', 'y', or 'z'. Default is 'y'.
        """
        local_rotations, local_positions, _, _, _, _ = self.get_data()

        angle_rad = np.radians(angle_degrees)
        half_angle = angle_rad / 2.0

        axis = axis.lower()
        if axis == "x":
            q = np.array([np.cos(half_angle), np.sin(half_angle), 0.0, 0.0], dtype=np.float32)
        elif axis == "y":
            q = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], dtype=np.float32)
        elif axis == "z":
            q = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")

        global_rotation = np.tile(q, (local_rotations.shape[0], 1))

        local_rotations[:, 0, :] = quat.mul(global_rotation, local_rotations[:, 0, :])
        local_positions[:, 0, :] = quat.mul_vec(global_rotation, local_positions[:, 0, :])

        self.set_data(local_rotations, local_positions)

    def _generate_mirror_mapping(self):
        """
        Generate a mirror mapping from joint names by detecting left/right
        symmetry patterns such as 'Left'/'Right', '_L'/'_R', 'L_'/'R_'.

        Returns
        -------
        mapping : np.ndarray[n_joints]
            Array where mapping[i] is the index of the mirrored joint for joint i.
            Center joints map to themselves.

        Examples
        --------
        For a 25-joint skeleton with standard naming, this would produce
        a mapping equivalent to:

        >>> # [0,1,21,22,23,24,6,7,8,9,17,18,19,20,14,15,16,10,11,12,13,2,3,4,5]
        >>> # Root->Root, Hips->Hips, LeftUpLeg<->RightUpLeg, etc.
        """
        names = list(self.data["names"])
        n_joints = len(names)
        mapping = np.arange(n_joints)

        # Pairs of (left_pattern, right_pattern) to check in joint names
        patterns = [
            ("Left", "Right"),
            ("left", "right"),
            ("_L", "_R"),
            ("_l", "_r"),
            ("L_", "R_"),
            ("l_", "r_"),
        ]

        name_to_idx = {name: i for i, name in enumerate(names)}

        for i, name in enumerate(names):
            if mapping[i] != i:
                # Already mapped by a previous pair
                continue
            for left_pat, right_pat in patterns:
                if left_pat in name:
                    mirror_name = name.replace(left_pat, right_pat)
                    if mirror_name in name_to_idx:
                        j = name_to_idx[mirror_name]
                        mapping[i] = j
                        mapping[j] = i
                        break
                elif right_pat in name:
                    mirror_name = name.replace(right_pat, left_pat)
                    if mirror_name in name_to_idx:
                        j = name_to_idx[mirror_name]
                        mapping[i] = j
                        mapping[j] = i
                        break

        return mapping

    def mirror(self, joints_mapping: np.ndarray = None, mode: str = "symmetry", axis: str = "X"):
        """
        Mirror the animation along the specified axis.

        Parameters
        ----------
        joints_mapping : np.ndarray, optional
            Array where joints_mapping[i] is the index of the mirrored joint
            for joint i. If None and mode is 'symmetry', automatically generates
            the mapping from joint names using left/right pattern matching.
        mode : str, optional
            Mirroring mode: 'symmetry', 'all', or 'positions'. Default is 'symmetry'.
        axis : str, optional
            Axis to mirror along: 'X', 'Y', or 'Z'. Default is 'X'.
        """
        local_rotations, local_positions, parents, offsets, end_sites, _ = self.get_data()
        global_translation = local_positions[:, 0, :].copy()

        if joints_mapping is None and mode == "symmetry":
            joints_mapping = self._generate_mirror_mapping()

        mirrored_rotations, mirrored_translation, mirrored_offsets, mirrored_end_sites = (
            skeleton.mirror(
                local_rotations=local_rotations,
                global_translation=global_translation,
                parents=parents,
                offsets=offsets,
                end_sites=end_sites,
                joints_mapping=joints_mapping,
                mode=mode,
                axis=axis,
            )
        )

        mirrored_positions = local_positions.copy()
        mirrored_positions[:, 0, :] = mirrored_translation

        self.set_data(mirrored_rotations, mirrored_positions)

    def copy_joint_order(self, reference_bvh):
        """
        Reorder joints to match the joint order of a reference BVH.
        Both BVHs must have the same set of joint names.

        Parameters
        ----------
        reference_bvh : BVH
            Reference BVH whose joint order will be copied.
        """
        ref_names = set(reference_bvh.data["names"])
        self_names = set(self.data["names"])

        if ref_names != self_names:
            missing_in_self = ref_names - self_names
            extra_in_self = self_names - ref_names
            error_msg = "The two BVH files do not have the same set of joints.\n"
            if missing_in_self:
                error_msg += f"  Missing in this BVH: {sorted(missing_in_self)}\n"
            if extra_in_self:
                error_msg += f"  Extra in this BVH (not in reference): {sorted(extra_in_self)}\n"
            raise ValueError(error_msg)

        if np.array_equal(self.data["names"], reference_bvh.data["names"]):
            return  # Already in the same order

        # Build the reorder mapping
        order = []
        for name in self.data["names"]:
            ref_idx = int(np.where(reference_bvh.data["names"] == name)[0][0])
            order.append(ref_idx)

        self.set_order_joints(order)

    def _save_joint(self, f, data, tab, i, joint_order):
        joint_order.append(i)

        f.write("%sJOINT %s\n" % (tab, data["names"][i]))
        f.write("%s{\n" % tab)
        tab += "\t"

        f.write(
            "%sOFFSET %f %f %f\n"
            % (
                tab,
                data["offsets"][i, 0],
                data["offsets"][i, 1],
                data["offsets"][i, 2],
            )
        )

        if data["channels"][i] == 6:  # joint has position and rotation channels
            f.write(
                "%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n"
                % (
                    tab,
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 0]],
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 1]],
                    self.inv_bvh_rot_map[self.data["rot_order"][0, 2]],
                )
            )
        else:  # joint has only rotation channels
            f.write(
                "%sCHANNELS 3 %s %s %s\n"
                % (
                    tab,
                    self.inv_bvh_rot_map[data["rot_order"][i, 0]],
                    self.inv_bvh_rot_map[data["rot_order"][i, 1]],
                    self.inv_bvh_rot_map[data["rot_order"][i, 2]],
                )
            )

        is_end_site = True

        for j in range(len(data["parents"])):
            if data["parents"][j] == i:
                tab = self._save_joint(f, data, tab, j, joint_order)
                is_end_site = False

        if is_end_site:
            f.write("%sEnd Site\n" % tab)
            f.write("%s{\n" % tab)
            tab += "\t"
            try:
                end_site_data = data["end_sites"][np.where(data["end_sites_parents"] == i)[0][0]]
            except ValueError:
                end_site_data = np.zeros(3)
            except KeyError:
                end_site_data = np.zeros(3)
            except IndexError:
                end_site_data = np.zeros(3)
            f.write("%sOFFSET %f %f %f\n" % (tab, end_site_data[0], end_site_data[1], end_site_data[2]))
            tab = tab[:-1]
            f.write("%s}\n" % tab)

        tab = tab[:-1]
        f.write("%s}\n" % tab)

        return tab
