import numpy as np
import pymotion.rotations.quat as quat
import copy


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

    def load(self, filename: str):
        """
        Reads a BVH file and returns a dictionary with the data.

        Parameters
        ----------
        filename : str
            path to the file.

        Results
        ------
        self.data : dict
            dictionary with the data.
            ["names"] : list[str]
                ith-element contain the name of ith-joint.
            ["offsets"] : np.array[n_joints, 3]
                ith-element contain the offset of ith-joint wrt. its parent joint.
            ["end_sites"] : np.array[n_end_sites, 3]
                ith-element contain the offset of ith-end-site wrt. its parent joint.
            ["end_sites_parents"] : list[int]
                ith-element contain the joint parent of the ith end-site.
            ["parents"] : list[int]
                ith-element contain the parent of the ith joint.
            ["rot_order"] : np.array[n_joints, 3]
                order per channel of the rotations. The order is 'x', 'y' or 'z'.
            ["positions"] : np.array[n_frames, n_joints, 3]
                local positions.
            ["rotations"] : np.array[n_frames, n_joints, 3]
                local rotations in euler angles with the order specified in rot_order.
            ["frame_time"] : float
                time between two frames in seconds.
        """
        f = open(filename, "r")

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
                    number_frames = int(line.split()[1])
                    offsets = np.array(offsets)
                    end_sites = np.array(end_sites)
                    rot_order = np.array(rot_order)
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
            "names": names,
            "offsets": offsets,
            "end_sites": end_sites,
            "end_sites_parents": end_sites_parents,
            "parents": parents,
            "rot_order": rot_order,
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
            if data == None, then self.data is used.
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
                if self.data["parents"][i] == 0:
                    tab = self._save_joint(f, self.data, tab, i, joint_order)

            tab = tab[:-1]
            f.write("%s}\n" % tab)

            f.write("%sMOTION\n" % tab)
            f.write("%sFrames: %d\n" % (tab, self.data["positions"].shape[0]))
            f.write("%sFrame Time: %f\n" % (tab, self.data["frame_time"]))

            for i in range(self.data["positions"].shape[0]):
                for j in joint_order:
                    if j == 0:  # root
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

        self.data["names"] = [self.data["names"][reverse_order[i]] for i in range(len(order))]
        self.data["offsets"] = self.data["offsets"][reverse_order]
        self.data["end_sites_parents"] = [order[j] for j in self.data["end_sites_parents"]]
        self.data["parents"][0] = 0
        self.data["parents"] = [order[self.data["parents"][reverse_order[i]]] for i in range(len(order))]
        self.data["parents"][0] = None
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
        keep_joints = [i for i in range(len(self.data["names"])) if i not in delete_joints]
        new_to_old = dict(enumerate(keep_joints))
        old_to_new = dict((v, k) for k, v in new_to_old.items())

        # Update transforms for remaining joints
        rots, pos, parents, offsets, end_sites, end_sites_parents = self.get_data()
        new_rots = rots[:, keep_joints, :]
        new_pos = pos[:, keep_joints, :]
        new_offsets = offsets[keep_joints, :]
        for j in keep_joints:
            while parents[j] not in keep_joints:
                p = parents[j]
                new_rots[:, old_to_new[j], :] = quat.mul(rots[:, p, :], new_rots[:, old_to_new[j], :])
                new_pos[:, old_to_new[j], :] = (
                    quat.mul_vec(rots[:, p, :], new_pos[:, old_to_new[j], :]) + pos[:, p, :]
                )
                new_offsets[old_to_new[j]] = (
                    quat.mul_vec(rots[0, p, :], new_offsets[old_to_new[j]]) + offsets[p]
                )
                parents[j] = parents[p]

        # Update parent indices for remaining joints
        new_parents = [0] * len(keep_joints)
        for i, p in enumerate(parents):
            if i in keep_joints:
                new_parents[old_to_new[i]] = old_to_new[p]
        new_parents[0] = None

        # Update end_sites_parents to reflect the removal of joints
        updated_end_sites_parents = [
            old_to_new[es_parent] for es_parent in end_sites_parents if es_parent in old_to_new
        ]
        # Remove end sites associated with deleted joints, if necessary
        valid_end_sites_indices = [
            i for i, es_parent in enumerate(end_sites_parents) if es_parent not in delete_joints
        ]

        # Update data
        self.data["names"] = [self.data["names"][i] for i in keep_joints]
        self.data["positions"] = new_pos
        self.data["rotations"] = np.degrees(
            quat.to_euler(
                new_rots, order=np.tile(self.data["rot_order"][..., keep_joints, :], (rots.shape[0], 1, 1))
            )
        )
        self.data["parents"] = new_parents
        self.data["offsets"] = new_offsets
        self.data["end_sites"] = end_sites[valid_end_sites_indices]
        self.data["end_sites_parents"] = updated_end_sites_parents

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
        parents : list[int]
            ith-element contain the parent of the ith joint.
        offsets : np.array[n_joints, 3]
            ith-element contain the offset of ith-joint wrt. its parent joint.
        end_sites : np.array[n_joints, 3]
            ith-element contain the offset of ith-end-site wrt. its parent joint.
        end_sites_parents : list[int]
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
        parents[0] = 0  # BVH sets root as None
        offsets = self.data["offsets"]
        end_sites = self.data["end_sites"]
        end_sites_parents = self.data["end_sites_parents"]
        return rots, pos, np.array(parents), offsets, end_sites, end_sites_parents

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
        parents : list[int]
            ith-element contain the parent of the ith joint.
        offsets : np.array[n_joints, 3]
            ith-element contain the offset of ith-joint wrt. its parent joint.
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
        self.data["parents"][0] = None  # BVH sets root as None

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
                end_site_data = data["end_sites"][data["end_sites_parents"].index(i)]
            except ValueError:
                end_site_data = np.zeros(3)
            f.write("%sOFFSET %f %f %f\n" % (tab, end_site_data[0], end_site_data[1], end_site_data[2]))
            tab = tab[:-1]
            f.write("%s}\n" % tab)

        tab = tab[:-1]
        f.write("%s}\n" % tab)

        return tab
