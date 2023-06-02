import socket
import struct
from threading import Thread
import bpy
import mathutils


class ModalOperator(bpy.types.Operator):
    """
    Get data from pymotion and render it in blender

    Types messages:
        1: render points
        2: render orientations

    Usage:
        - Open the Text Editor window in Blender
        - Open this file
        - Run this file
        - Run a pymotion script that sends data to this server
        - Press ESC to stop the server
    """

    bl_idname = "object.modal_operator"
    bl_label = "PyMotion Communication Server"

    def modal(self, context, event):
        if event.type in {"ESC"}:
            self.conn.close()
            self.s.close()
            self.finish_thread = True
            self.thread.join()
            return {"CANCELLED"}

        # Wait for message code
        if self.message_id is not None and self.message_data is not None:
            # Clear all objects
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
            # Render message
            if self.message_id == 1:
                self.render_points()
            elif self.message_id == 2:
                self.render_orientations()
            else:
                raise Exception("Unknown message id {}".format(self.message_id))
            self.message_data = None
            self.message_id = None

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        # invoked once when operator is called
        self.conn = None
        self.s = None
        self.init_socket()
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def init_socket(self):
        if self.conn is not None:
            self.conn.close()

        if self.s is not None:
            self.s.close()

        self.host = "127.0.0.1"
        self.port = 2222
        # open a socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(
            (self.host, self.port)
        )  # associate socket with interface en port number
        self.s.listen()
        print("socket listening...")
        (
            self.conn,
            self.addr,
        ) = (
            self.s.accept()
        )  # blocks and waits for a connection, conn is a new socket, addr is (host, port)
        print("Connected by {}".format(self.addr))
        # Thread to receive messages
        self.thread = Thread(target=self.receive_messages, daemon=True)
        self.thread.start()

    def receive_messages(self):
        self.finish_thread = False
        self.message_id = None
        self.message_data = None
        while not self.finish_thread:
            message_id = self.receive_message_id()
            if message_id is None:
                self.finish_thread = True
                continue
            self.message_id = message_id
            print("Received message id {}".format(self.message_id))
            self.message_data = self.receive_data()

    def receive_message_id(self):
        id = self.conn.recv(4)
        if not id:
            return None  # if empty byte object b'' is returned -> client closed the connection
        self.conn.sendall(struct.pack("<i", 1))
        return struct.unpack("<i", id)[0]

    def receive_data(self):
        size_data = self.conn.recv(4)
        size = struct.unpack("<i", size_data)[0]
        self.conn.sendall(struct.pack("<i", 1))
        data = self.conn.recv(4 * size)
        floats = []
        for i in range(size):
            floats.append(struct.unpack("<f", data[i * 4 : i * 4 + 4])[0])
        return floats

    def render_points(self):
        # reshape data
        positions = []
        for i in range(0, len(self.message_data), 3):
            pos = self.message_data[i : i + 3]
            positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))
        # create objects
        for i, pos in enumerate(positions):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=pos)
            bpy.context.object.name = "point_{}".format(i)

    def render_orientations(self):
        # reshape data
        quaternions = []
        positions = []
        for i in range(0, len(self.message_data), 7):  # 3 (positions) + 4 (quaternions)
            pos = self.message_data[i : i + 3]
            quat = self.message_data[i + 3 : i + 7]
            quaternions.append(
                mathutils.Quaternion([quat[0], quat[1], quat[2], quat[3]])
            )
            positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))
        # create objects
        for i, quat in enumerate(quaternions):
            bpy.ops.object.empty_add(type="ARROWS", location=positions[i])
            bpy.context.object.name = "orientation_{}".format(i)
            bpy.context.object.rotation_mode = "QUATERNION"
            bpy.context.object.rotation_quaternion = quat


def menu_func(self, context):
    self.layout.operator(ModalOperator.bl_idname, text=ModalOperator.bl_label)


# Register and add to the "view" menu (required to also use F3 search "PyMotion Communication Server" for quick access).
def register():
    bpy.utils.register_class(ModalOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)


if __name__ == "__main__":
    register()
    bpy.ops.object.modal_operator("INVOKE_DEFAULT")
