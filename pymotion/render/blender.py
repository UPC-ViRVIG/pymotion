import socket
import struct
import numpy as np


class BlenderConnection:
    """
    Types messages:
        1: render points
        2: render orientations

    Example
    -------
        >>> conn = BlenderConnection("127.0.0.1", 2222)
        >>> conn.render_points(np.array([[0, 0, 0], [1, 2, 3]]))
        >>> conn.close()
    """

    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        # open a socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(
            (self.host, self.port)
        )  # associate socket with interface in port number
        print("Connected to {}:{}".format(self.host, self.port))

    def render_points(self, points: np.array) -> None:
        """
        Parameters
        ----------
            points : np.array[..., 3]
        """
        assert points.shape[-1] == 3
        flatten_points = points.flatten()
        message_id = 1
        self._send_message_code(message_id)
        number_points = flatten_points.shape[0]
        self._send_data(number_points, flatten_points)

    def render_orientations(self, points: np.array, orientations: np.array) -> None:
        """
        Parameters
        ----------
            points (Optional) : np.array[..., 3]
            orientations : np.array[..., 4]
        """
        if points is None:
            points = np.zeros(orientations.shape[:-1] + (3,))
        assert points.shape[-1] == 3
        assert orientations.shape[-1] == 4
        data = np.concatenate((points, orientations), axis=-1)
        flatten_data = data.flatten()
        message_id = 2
        self._send_message_code(message_id)
        number_orientations = flatten_data.shape[0]
        self._send_data(number_orientations, flatten_data)

    def _send_message_code(self, message_id: int) -> None:
        self.s.sendall(struct.pack("<i", message_id))
        # wait for confirmation
        ack = self.s.recv(4)
        if struct.unpack("<i", ack)[0] != 1:
            raise Exception("Some error occurred during communication")

    def _send_data(self, size: int, data: np.array) -> None:
        # send size
        self.s.sendall(struct.pack("<i", size))
        # wait for confirmation
        ack = self.s.recv(4)
        if struct.unpack("<i", ack)[0] != 1:
            raise Exception("Some error occurred during communication")
        # send data
        self.s.sendall(struct.pack("<{}f".format(len(data)), *(data)))

    def close(self):
        self.s.close()
