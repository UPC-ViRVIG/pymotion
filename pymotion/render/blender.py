# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Union

import numpy as np
from pymotion.io.bvh import BVH

pymotion_blender_script = os.path.join(os.path.dirname(__file__), "internal", "pymotion_blender.py")


class BlenderAutoStarter:
    def __init__(self, blender_executable_path=None):
        self.blender_executable = self._find_blender_executable(blender_executable_path)
        if not self.blender_executable:
            raise FileNotFoundError("Blender executable not found. Please install Blender.")

    def _find_blender_executable(self, blender_executable_path=None):
        possible_executables = []
        if blender_executable_path and os.path.exists(blender_executable_path):
            possible_executables.append(blender_executable_path)
        else:
            if os.name == "nt":  # Windows
                if os.path.isdir("C:\\Program Files\\Blender Foundation\\"):
                    for dir in os.listdir("C:\\Program Files\\Blender Foundation\\"):
                        if "blender" in dir.lower():
                            possible_executables.append(
                                f"C:\\Program Files\\Blender Foundation\\{dir}\\blender.exe"
                            )
                possible_executables.append(
                    [
                        "blender.exe",  # In PATH
                    ]
                )
            else:  # Linux/macOS
                possible_executables.extend(
                    [
                        "/usr/bin/blender",  # Common Linux path
                        "/usr/local/bin/blender",  # Common macOS path
                        "blender",  # In PATH
                    ]
                )

        for exe in possible_executables:
            if shutil.which(exe):
                return shutil.which(exe)
        return None

    def start_blender(self, blender_script_path, port):
        command = [
            self.blender_executable,
            "-P",
            blender_script_path,  # Run script on startup
            "--",  # Separator for Blender script arguments
            str(port),  # Add port number as a string argument
        ]
        try:
            subprocess.Popen(command)  # Non-blocking call
            print(f"[DEBUG PYTHON] Blender started in background with script: {blender_script_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to start Blender: {e}")

    def start_and_connect_blender(self, port):
        self.start_blender(pymotion_blender_script, port)


class BlenderConnection:
    """
    Types messages:
        0: clear scene
        1: render points
        2: render orientations
        3: render BVH
        4: render checkerboard floor
        5: render points timeline
        6: set up rendering

    Example
    -------
        >>> with BlenderConnection() as conn:
        >>>    conn.clear_scene()
        >>>    conn.render_checkerboard_floor()
        >>>    conn.render_points(
        >>>        np.array([[0, -3, 0], [1, 2, 3]]), np.array([[0, 0, 1], [0, 1, 0]]), radius=np.array([[0.25], [0.05]])
        >>>    )
        >>>    conn.render_orientations(
        >>>        np.array([[1, 0, 0, 0], [np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0, 0]]),
        >>>        np.array([[0, -3, 0], [1, 2, 3]]),
        >>>        scale=np.array([[0.5], [0.25]]),
        >>>    )
        >>>    # BVH files can be rendered directly from file path
        >>>    path = "test.bvh"
        >>>    conn.render_bvh_from_path(
        >>>        path,
        >>>        np.array([0, 0, 1]),
        >>>        end_joints=["RightWrist", "LeftWrist", "RightToe", "LeftToe", "Head"],
        >>>    )
        >>>    # or by using a BVH object
        >>>    bvh = BVH()
        >>>    path = "test2.bvh"
        >>>    bvh.load(path)
        >>>    conn.render_bvh(
        >>>        bvh, np.array([0, 1, 0]), end_joints=["RightWrist", "LeftWrist", "RightToe", "LeftToe", "Head"]
        >>>    )
    """

    def __init__(self, port: int = 2222, retries: int = 10, delay: float = 2.0) -> None:
        self.host = "127.0.0.1"
        self.port = port
        self.blender_starter = BlenderAutoStarter()
        self.s = None

        try:
            self._connect_socket()  # Try to connect first, assuming Blender might be running
        except ConnectionRefusedError:
            print(
                "[DEBUG PYTHON] Connection refused. Assuming Blender is not running.",
                file=sys.stderr,
            )
            print("[DEBUG PYTHON] Starting Blender...")
            self.blender_starter.start_and_connect_blender(self.port)

            # Retry connecting after launching Blender
            for i in range(retries):
                try:
                    print(f"[DEBUG PYTHON] Connection attempt {i + 1}/{retries}...")
                    self._connect_socket()
                    # If connection succeeds, break the loop
                    return
                except ConnectionRefusedError:
                    if i < retries - 1:
                        time.sleep(delay)
                    else:
                        # If all retries fail, raise the exception
                        print(
                            "[DEBUG PYTHON] Could not connect to Blender after multiple attempts.",
                            file=sys.stderr,
                        )
                        raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _connect_socket(self):
        if self.s is None:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))  # associate socket with interface in port number
        print(f"[DEBUG PYTHON] Connected to {self.host}:{self.port}")

    def clear_scene(self) -> None:
        message_id = 0
        self._send_message_code(message_id)

    def setup_rendering(self) -> None:
        message_id = 6
        self._send_message_code(message_id)

    def render_points(
        self,
        points: np.ndarray,
        color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        radius: np.ndarray = np.array([0.1]),
    ) -> None:
        """
        Parameters
        ----------
            points : np.ndarray[..., 3]
            color (Optional) : np.ndarray[3] or np.ndarray[..., 3]
            radius (Optional) : np.ndarray[1] or np.ndarray[..., 1]
        """
        if points.shape[-1] != 3:
            raise ValueError("Points must have shape [..., 3]")
        if color.shape[-1] != 3:
            raise ValueError("Color must have shape [3] or [..., 3]")
        if radius.shape[-1] != 1:
            raise ValueError("Radius must have shape [1] or [..., 1]")
        if color.ndim > 1 and points.shape != color.shape:
            raise ValueError("Color must have shape [3] or [..., 3] matching points shape")
        if radius.ndim > 1 and points.shape[:-1] != radius.shape[:-1]:
            raise ValueError("Radius must have shape [1] or [..., 1] matching points shape")

        if color.ndim == 1:
            color = np.tile(color, points.shape[:-1] + (1,))
        if radius.ndim == 1:
            radius = np.tile(radius, points.shape[:-1] + (1,))

        flatten_points = points.flatten()
        flatten_color = color.flatten()
        flatten_radius = radius.flatten()
        message_id = 1
        self._send_message_code(message_id)
        self._send_data(flatten_points, flatten_color, flatten_radius)

    def render_points_timeline(
        self,
        points: np.ndarray,
        color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        radius: np.ndarray = np.array([0.1]),
    ) -> None:
        """
        Parameters
        ----------
            points : np.ndarray[frames, ..., 3]
            color (Optional) : np.ndarray[3] or np.ndarray[..., 3]
            radius (Optional) : np.ndarray[1] or np.ndarray[..., 1]
        """
        if points.ndim < 2:
            raise ValueError("Points must have shape [frames, ..., 3]")
        if points.shape[-1] != 3:
            raise ValueError("Points must have shape [frames, ..., 3]")
        if color.shape[-1] != 3:
            raise ValueError("Color must have shape [3] or [..., 3]")
        if radius.shape[-1] != 1:
            raise ValueError("Radius must have shape [1] or [..., 1]")
        if color.ndim > 1 and points.shape[1:] != color.shape:
            raise ValueError("Color must have shape [3] or [..., 3] matching points shape")
        if radius.ndim > 1 and points.shape[1:-1] != radius.shape[:-1]:
            raise ValueError("Radius must have shape [1] or [..., 1] matching points shape")

        if color.ndim == 1 and points.ndim > 2:
            color = np.tile(color, points.shape[1:-1] + (1,))
        if radius.ndim == 1 and points.ndim > 2:
            radius = np.tile(radius, points.shape[1:-1] + (1,))

        frames = np.array([points.shape[0]])
        flatten_points = points.flatten()
        flatten_points = np.concatenate((frames, flatten_points), axis=0)
        flatten_color = color.flatten()
        flatten_radius = radius.flatten()
        message_id = 5
        self._send_message_code(message_id)
        self._send_data(flatten_points, flatten_color, flatten_radius)

    def render_orientations(
        self,
        orientations: np.ndarray,
        points: np.ndarray = None,
        scale: np.ndarray = np.array([1.0]),
    ) -> None:
        """
        Parameters
        ----------
            orientations : np.ndarray[..., 4]
            points (Optional) : np.ndarray[..., 3]
            scale (Optional) : np.ndarray[1] or np.ndarray[..., 1]
        """

        if orientations.shape[-1] != 4:
            raise ValueError("Orientations must have shape [..., 4]")
        if points is not None and points.shape[-1] != 3:
            raise ValueError("Points must have shape [..., 3]")
        if scale.shape[-1] != 1:
            raise ValueError("Scale must have shape [1] or [..., 1]")
        if points is not None and points.shape[:-1] != orientations.shape[:-1]:
            raise ValueError("Points must have shape [..., 3] matching orientations")
        if scale.ndim > 1 and orientations.shape[:-1] != scale.shape[:-1]:
            raise ValueError("Scale must have shape [1] or [..., 1] matching orientations shape")

        if points is None:
            points = np.zeros(orientations.shape[:-1] + (3,))
        if scale.ndim == 1:
            scale = np.tile(scale, orientations.shape[:-1] + (1,))

        data = np.concatenate((points, orientations), axis=-1)
        flatten_data = data.flatten()
        flatten_scale = scale.flatten()
        message_id = 2
        self._send_message_code(message_id)
        self._send_data(flatten_data, None, flatten_scale)

    def render_bvh(
        self,
        bvh: BVH,
        color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        end_joints: list[str] = None,
        axis_forward: str = "-Z",
        axis_up: str = "Y",
    ):
        """
        Parameters
        ----------
            bvh : BVH
            color (Optional) : np.ndarray[3]
            end_joints (Optional) : List[str]
            axis_forward (Optional) : str
                Forward axis of the BVH file (can be "-X", "X", "-Y", "Y", "-Z", "Z")
            axis_up (Optional) : str
                Up axis of the BVH file (can be "-X", "X", "-Y", "Y", "-Z", "Z")
        """
        if isinstance(bvh, BVH) is False:
            raise ValueError("BVH object expected")
        if color.shape != (3,):
            raise ValueError("Color must have shape [3]")
        if end_joints is not None and not all(isinstance(joint, str) for joint in end_joints):
            raise ValueError("End joints must be a list of strings")
        axis_candidates = ["-X", "X", "-Y", "Y", "-Z", "Z"]
        if axis_forward not in axis_candidates:
            raise ValueError("Axis forward must be one of -X, X, -Y, Y, -Z, Z")
        if axis_up not in axis_candidates:
            raise ValueError("Axis up must be one of -X, X, -Y, Y, -Z, Z")

        # get a temporal path
        filename = bvh.data["filename"]
        with tempfile.NamedTemporaryFile(prefix=filename, suffix=".bvh", delete=False) as f:
            bvh.save(f.name)
            self.render_bvh_from_path(f.name, color, end_joints, axis_forward, axis_up, delete_after=True)

    def render_bvh_from_path(
        self,
        bvh_path: str,
        color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        end_joints: list[str] = None,
        axis_forward: str = "-Z",
        axis_up: str = "Y",
        delete_after: bool = False,
    ):
        """
        Parameters
        ----------
            bvh_path : str
            color (Optional) : np.ndarray[3]
            end_joints (Optional) : List[str]
            axis_forward (Optional) : str
                Forward axis of the BVH file (can be "-X", "X", "-Y", "Y", "-Z", "Z")
            axis_up (Optional) : str
                Up axis of the BVH file (can be "-X", "X", "-Y", "Y", "-Z", "Z")
            delete_after (Optional) : bool
                Whether to delete the BVH file after rendering
        """

        if isinstance(bvh_path, str) is False:
            raise ValueError("BVH path must be a string")
        if color.shape != (3,):
            raise ValueError("Color must have shape [3]")
        if end_joints is not None and not all(isinstance(joint, str) for joint in end_joints):
            raise ValueError("End joints must be a list of strings")
        axis_candidates = ["-X", "X", "-Y", "Y", "-Z", "Z"]
        if axis_forward not in axis_candidates:
            raise ValueError("Axis forward must be one of -X, X, -Y, Y, -Z, Z")
        if axis_up not in axis_candidates:
            raise ValueError("Axis up must be one of -X, X, -Y, Y, -Z, Z")

        message_id = 3
        self._send_message_code(message_id)
        bvh_path += ";".join(end_joints) if end_joints else ""
        bvh_path += f";{axis_forward};{axis_up}"
        self._send_data(bvh_path, color, np.array([1.0]) if delete_after else np.array([0.0]))

    def render_checkerboard_floor(
        self,
        plane_size: float = 40.0,
        checker_size: float = 0.25,
        color1: np.ndarray = np.array([0.8, 0.9, 1.0]),
        color2: np.ndarray = np.array([0.5, 0.55, 0.6]),
    ):
        """
        Parameters
        ----------
            plane_size (Optional) : int
                Size of the plane
            checker_size (Optional) : float
                Size of each checker
            color1 (Optional) : np.ndarray[3]
                Color of the first checker
            color2 (Optional) : np.ndarray[3]
                Color of the second checker
        """
        if not isinstance(plane_size, float):
            raise ValueError("Plane size must be a float")
        if not isinstance(checker_size, float):
            raise ValueError("Checker size must be a float")
        if color1.shape != (3,):
            raise ValueError("Color1 must have shape [3]")
        if color2.shape != (3,):
            raise ValueError("Color2 must have shape [3]")

        message_id = 4
        data = np.array([plane_size, checker_size, *color1, *color2])
        self._send_message_code(message_id)
        self._send_data(data)

    def _send_message_code(self, message_id: int) -> None:
        # print(f"[DEBUG PYTHON] Sending message code {message_id}")
        self.s.sendall(struct.pack("<i", message_id))
        # print("[DEBUG PYTHON] Sent message code")
        # wait for confirmation
        ack = self.s.recv(4)
        # print("[DEBUG PYTHON] Received ack")
        if not ack or struct.unpack("<i", ack)[0] != 1:
            raise ConnectionAbortedError("Communication with Blender failed. It might have crashed.")
        # print("[DEBUG PYTHON] End of send_message_code")

    def _send_data(
        self,
        data: Union[np.ndarray, str],
        color: np.ndarray = None,
        scale: np.ndarray = None,
    ) -> None:

        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
            data_len = len(data_bytes)
            data_format_char = "s"
        elif isinstance(data, np.ndarray):
            data_len = len(data)
            data_format_char = "f"
        else:
            raise TypeError("Data must be either a string or a NumPy array.")

        # send size
        self.s.sendall(struct.pack("<i", data_len))
        self.s.sendall(struct.pack("<i", 0 if color is None else len(color)))
        self.s.sendall(struct.pack("<i", 0 if scale is None else len(scale)))

        # send if data is string (0) or float array (1)
        self.s.sendall(struct.pack("<i", 0 if data_format_char == "s" else 1))

        # wait for confirmation
        ack = self.s.recv(4)
        if not ack or struct.unpack("<i", ack)[0] != 1:
            raise ConnectionAbortedError("Communication with Blender failed. It might have crashed.")

        # send data
        if data_format_char == "s":
            # Send string data as bytes
            self.s.sendall(struct.pack(f"<{data_len}{data_format_char}", data_bytes))
        elif data_format_char == "f":
            # Send float array data
            self.s.sendall(struct.pack(f"<{data_len}{data_format_char}", *data))

        if color is not None:
            self.s.sendall(struct.pack(f"<{len(color)}f", *(color)))
        if scale is not None:
            self.s.sendall(struct.pack(f"<{len(scale)}f", *(scale)))

    def close(self):
        self.s.close()
        self.s = None
