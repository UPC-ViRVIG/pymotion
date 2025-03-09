import os
import socket
import struct
import sys
import time
import bpy
import bmesh
import mathutils
from threading import Thread

# Get command line arguments
argv = sys.argv

# Blender Python scripts usually start with blender executable path and script path as the first two arguments.
# Arguments passed after "--" start from index where "--" was in the command.
try:
    script_arg_index = argv.index("--") + 1
except ValueError:
    print("Error: No '--' argument found when starting Blender.")
    port_number = 2222
else:
    if script_arg_index < len(argv):
        try:
            port_number_str = argv[script_arg_index]
            port_number = int(port_number_str)
        except ValueError:
            print(f"Error: Invalid port number argument: '{argv[script_arg_index]}'. Expected an integer.")
            port_number = 2222
    else:
        print("Error: Port number argument missing after '--'.")
        port_number = 2222

HOST = "127.0.0.1"
PORT = port_number
CONN = None
ADDR = None
SOCKET = None
MESSAGE_ID = None
MESSAGE_DATA = None
MESSAGE_COLOR = None
MESSAGE_SCALE = None
FINISH_THREAD = False
RECEIVE_THREAD = None
MATERIAL_CACHE = {}


def init_socket():
    global SOCKET, CONN, ADDR
    if CONN:
        CONN.close()
        CONN = None
    if SOCKET:
        SOCKET.close()
        SOCKET = None

    SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        SOCKET.bind((HOST, PORT))
        SOCKET.listen()
        CONN, ADDR = SOCKET.accept()
        return True  # Socket initialization successful
    except Exception as e:
        print(f"Socket initialization error: {e}")
        return False  # Socket initialization failed


def receive_messages():
    global FINISH_THREAD, MESSAGE_ID, MESSAGE_DATA, MESSAGE_COLOR, MESSAGE_SCALE, CONN
    MESSAGE_ID = None
    MESSAGE_DATA = None
    MESSAGE_COLOR = None
    MESSAGE_SCALE = None
    while not FINISH_THREAD:
        if CONN is None:  # Check if connection is established
            if not init_socket():  # Try to re-initialize if no connection
                time.sleep(1)  # Add a small delay to avoid busy-looping
                continue  # Keep looping to re-attempt init_socket
            else:
                continue  # Wait for connection to be established

        message_id_val = receive_message_id()
        if message_id_val is None:
            CONN = None  # Reset connection for re-accepting
            continue  # Go back to check for new connections/stop thread

        MESSAGE_ID = message_id_val
        if MESSAGE_ID != 0:
            MESSAGE_DATA, MESSAGE_COLOR, MESSAGE_SCALE = receive_data()

    if CONN:
        CONN.close()
    if SOCKET:
        SOCKET.close()


def receive_message_id():
    global CONN, MESSAGE_ID
    try:
        if CONN is None:  # Check connection before receiving
            return None

        id_bytes = CONN.recv(4)
        if not id_bytes:
            return None  # if empty byte object b'' is returned -> client closed the connection

        while MESSAGE_ID is not None:
            time.sleep(0.1)  # Wait until MESSAGE_ID is processed

        CONN.sendall(struct.pack("<i", 1))  # Acknowledge ID reception
        return struct.unpack("<i", id_bytes)[0]
    except Exception as e:
        print(f"Error receiving message ID: {e}")
        return None


def receive_data():
    global CONN
    try:
        if CONN is None:  # Check connection before receiving
            return None

        size_data_bytes = CONN.recv(4)
        size_data = struct.unpack("<i", size_data_bytes)[0]
        size_color_bytes = CONN.recv(4)
        size_color = struct.unpack("<i", size_color_bytes)[0]
        size_scale_bytes = CONN.recv(4)
        size_scale = struct.unpack("<i", size_scale_bytes)[0]

        data_type_indicator_bytes = CONN.recv(4)
        data_type_indicator = struct.unpack("<i", data_type_indicator_bytes)[0]

        CONN.sendall(struct.pack("<i", 1))  # Acknowledge metadata reception

        # Data
        if data_type_indicator == 0:  # string
            data_bytes = CONN.recv(size_data)  # Receive string data
            data = data_bytes.decode("utf-8")
        elif data_type_indicator == 1:  # float array
            data_bytes = CONN.recv(4 * size_data)  # Receive float data (4 bytes per float)
            data = []
            for i in range(size_data):
                data.append(struct.unpack("<f", data_bytes[i * 4 : i * 4 + 4])[0])
        else:
            raise ValueError(f"Unknown data type indicator: {data_type_indicator}")

        # Color
        if size_color > 0:
            color_bytes = CONN.recv(4 * size_color)  # Receive float data (4 bytes per float)
            colors = []
            for i in range(size_color):
                colors.append(struct.unpack("<f", color_bytes[i * 4 : i * 4 + 4])[0])
        else:
            colors = None

        # Scale
        if size_scale > 0:
            scales_bytes = CONN.recv(4 * size_scale)  # Receive float data (4 bytes per float)
            scales = []
            for i in range(size_scale):
                scales.append(struct.unpack("<f", scales_bytes[i * 4 : i * 4 + 4])[0])
        else:
            scales = None

        return data, colors, scales
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def render_points():
    global MESSAGE_DATA, MESSAGE_COLOR, MESSAGE_SCALE
    if MESSAGE_DATA is None or MESSAGE_COLOR is None or MESSAGE_SCALE is None:
        raise ValueError("Render points: Data, color or scale is None.")

    positions = []
    for i in range(0, len(MESSAGE_DATA), 3):
        pos = MESSAGE_DATA[i : i + 3]
        positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))

    colors = []
    for i in range(0, len(MESSAGE_COLOR), 3):
        col = MESSAGE_COLOR[i : i + 3]
        colors.append((col[0], col[1], col[2]))

    radius = []
    for i in range(0, len(MESSAGE_SCALE), 1):
        rad = MESSAGE_SCALE[i]
        radius.append(rad)

    for i, pos in enumerate(positions):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius[i], location=pos)
        obj = bpy.context.object
        obj.name = f"point_{i}"
        material = get_material(colors[i])
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def render_orientations():
    global MESSAGE_DATA, MESSAGE_SCALE
    if MESSAGE_DATA is None or MESSAGE_SCALE is None:
        raise ValueError("Render orientations: Data or scale is None.")

    quaternions = []
    positions = []
    for i in range(0, len(MESSAGE_DATA), 7):  # 3 (positions) + 4 (quaternions)
        pos = MESSAGE_DATA[i : i + 3]
        quat = MESSAGE_DATA[i + 3 : i + 7]
        quaternions.append(mathutils.Quaternion([quat[0], quat[1], quat[2], quat[3]]))
        positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))

    scales = []
    for i in range(0, len(MESSAGE_SCALE), 1):
        scale = MESSAGE_SCALE[i]
        scales.append(scale)

    for i, quat in enumerate(quaternions):
        bpy.ops.object.empty_add(type="ARROWS", location=positions[i])
        obj = bpy.context.object
        obj.name = "orientation_{}".format(i)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat
        obj.scale = (scales[i], scales[i], scales[i])


def render_bvh():
    global MESSAGE_DATA, MESSAGE_COLOR, MESSAGE_SCALE
    if MESSAGE_DATA is None or MESSAGE_COLOR is None or MESSAGE_SCALE is None:
        raise ValueError("Render BVH: Data, color or scale is None.")

    data = MESSAGE_DATA.split(".bvh")
    if len(data) != 2:
        raise ValueError("Invalid BVH data format.")
    bvh_path = data[0] + ".bvh"
    data = data[1].split(";")
    end_joints = data[:-2]
    axis_forward = data[-2]
    axis_up = data[-1]
    color = (MESSAGE_COLOR[0], MESSAGE_COLOR[1], MESSAGE_COLOR[2])
    should_delete_file = MESSAGE_SCALE[0] == 1

    bpy.data.scenes["Scene"].frame_end = 1
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.import_anim.bvh(
        filepath=bvh_path,
        filter_glob="*.bvh",
        target="ARMATURE",
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
        use_cyclic=False,
        rotate_mode="NATIVE",
        axis_forward=axis_forward,
        axis_up=axis_up,
    )
    if should_delete_file:
        os.remove(bvh_path)

    generate_rig_representation(bpy.context.active_object, color, end_joints=end_joints)


def render_checkerboard_floor():
    global MESSAGE_DATA
    if MESSAGE_DATA is None:
        raise ValueError("Render checkerboard floor: Data is None.")

    plane_size = MESSAGE_DATA[0]
    checker_size = MESSAGE_DATA[1]
    color1 = (MESSAGE_DATA[2], MESSAGE_DATA[3], MESSAGE_DATA[4], 1.0)
    color2 = (MESSAGE_DATA[5], MESSAGE_DATA[6], MESSAGE_DATA[7], 1.0)

    create_checkerboard_plane(plane_size, checker_size, color1, color2)


def get_material(color_rgb):  # Function to get or create material
    global MATERIAL_CACHE

    if color_rgb in MATERIAL_CACHE:
        return MATERIAL_CACHE[color_rgb]  # Reuse existing material

    material = bpy.data.materials.new(name=f"PyMotionMat_{color_rgb}")  # Create new material
    material.use_nodes = True
    principled_bsdf = material.node_tree.nodes["Principled BSDF"]
    principled_bsdf.inputs["Base Color"].default_value = (
        color_rgb[0],
        color_rgb[1],
        color_rgb[2],
        1.0,
    )  # RGBA (alpha 1.0)
    MATERIAL_CACHE[color_rgb] = material  # Store in cache
    return material


def generate_rig_representation(armature_obj, color, end_joints=None):
    if armature_obj.type != "ARMATURE":
        print("Selected object is not an armature!")
        return

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.object.data.display_type = "STICK"
    bpy.context.object.data.pose_position = "REST"

    # Material
    material = get_material(color)

    # Create a new collection
    rig_collection = bpy.data.collections.new(name=armature_obj.name)
    bpy.context.scene.collection.children.link(rig_collection)

    bones = armature_obj.data.bones
    for bone in bones:
        head_location = armature_obj.matrix_world @ bone.head_local
        tail_location = armature_obj.matrix_world @ bone.tail_local

        sphere_head = create_sphere_at_location(head_location, 0.04, bone.name)
        sphere_head.data.materials.append(material)
        sphere_current_collection = sphere_head.users_collection[0]
        # Link the new object to the collection
        rig_collection.objects.link(sphere_head)
        # Then, unlink the object from the main scene collection to avoid duplicates
        sphere_current_collection.objects.unlink(sphere_head)
        setup_constraints(sphere_head, bone.name, armature_obj)

        if end_joints is not None and bone.name in end_joints:
            continue

        cylinder = create_cylinder_between_points(bone, head_location, tail_location, 0.02, bone.name)
        cylinder.data.materials.append(material)
        cylinder_current_collection = cylinder.users_collection[0]
        rig_collection.objects.link(cylinder)
        cylinder_current_collection.objects.unlink(cylinder)
        setup_constraints(cylinder, bone.name, armature_obj)

    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.context.object.data.pose_position = "POSE"


def create_sphere_at_location(location, radius=0.1, name="Sphere"):
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius, calc_uvs=True)
    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    return obj


def create_cylinder_between_points(bone, p1, p2, radius=0.2, name="Cylinder"):
    direction = p2 - p1
    length = direction.length

    # Create cylinder mesh and object
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=5, radius1=radius, radius2=radius, depth=length)
    bm.to_mesh(mesh)
    bm.free()

    # Position it
    obj.location = (p1 + p2) / 2
    rotation_difference = direction.rotation_difference(mathutils.Vector((0, 0, 1)))
    r = rotation_difference.to_euler()
    obj.rotation_euler = mathutils.Vector((-r.x, -r.y, -r.z))

    return obj


def setup_constraints(obj, target_bone_name, armature_object):
    constraint = obj.constraints.new(type="CHILD_OF")
    constraint.target = armature_object
    constraint.subtarget = target_bone_name


def create_checkerboard_plane(plane_size=2, checker_size=1, color1=(1, 1, 1, 1), color2=(0, 0, 0, 1)):
    """
    Create a plane with a checkerboard pattern in Blender.

    Parameters:
    - plane_size: The overall size of the plane.
    - checker_size: The size of each individual checker square.
    - color1 and color2: The colors for the checker pattern.
    """
    # Create a plane
    bpy.ops.mesh.primitive_plane_add(size=plane_size)
    plane = bpy.context.object

    # Create a new material and assign to plane
    mat = bpy.data.materials.new(name="Checker_Material")
    plane.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add a diffuse shader
    shader = nodes.new(type="ShaderNodeBsdfDiffuse")

    # Add a checker texture node and set its values
    checker_node = nodes.new(type="ShaderNodeTexChecker")
    checker_node.inputs["Scale"].default_value = plane_size / checker_size
    checker_node.inputs["Color1"].default_value = color1
    checker_node.inputs["Color2"].default_value = color2

    mat.node_tree.links.new(shader.inputs["Color"], checker_node.outputs["Color"])

    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(output.inputs["Surface"], shader.outputs["BSDF"])

    return plane


def check_messages():
    global MESSAGE_ID, MESSAGE_DATA, MESSAGE_COLOR, MESSAGE_SCALE
    if MESSAGE_ID is not None:
        if MESSAGE_ID == 0:
            clear_scene()
        elif MESSAGE_ID == 1:
            render_points()
        elif MESSAGE_ID == 2:
            render_orientations()
        elif MESSAGE_ID == 3:
            render_bvh()
        elif MESSAGE_ID == 4:
            render_checkerboard_floor()
        else:
            print(f"Unknown message id {MESSAGE_ID}")
        MESSAGE_ID = None
        MESSAGE_DATA = None
        MESSAGE_COLOR = None
        MESSAGE_SCALE = None
    return 0.1  # Timer interval (check for messages every 0.1 seconds)


def start_server():
    global FINISH_THREAD, RECEIVE_THREAD
    if not init_socket():  # Initialize socket and check for success
        print("Failed to initialize socket. Server not starting.")
        return

    FINISH_THREAD = False
    RECEIVE_THREAD = Thread(target=receive_messages, daemon=True)
    RECEIVE_THREAD.start()
    bpy.app.timers.register(check_messages)  # Start timer to periodically check for messages
    clear_scene()


def stop_server():
    global FINISH_THREAD, RECEIVE_THREAD
    FINISH_THREAD = True
    if RECEIVE_THREAD:
        RECEIVE_THREAD.join(timeout=2)  # Wait for thread to finish (with timeout)
        if RECEIVE_THREAD.is_alive():
            print("Warning: Receive thread did not terminate gracefully.")
    if CONN:
        CONN.close()
    if SOCKET:
        SOCKET.close()


class StartPyMotionServerOperator(bpy.types.Operator):
    bl_idname = "object.start_pymotion_server"
    bl_label = "Start PyMotion Server"

    def execute(self, context):
        start_server()
        return {"FINISHED"}


class StopPyMotionServerOperator(bpy.types.Operator):
    bl_idname = "object.stop_pymotion_server"
    bl_label = "Stop PyMotion Server"

    def execute(self, context):
        stop_server()
        return {"FINISHED"}


def menu_func(self, context):
    layout = self.layout
    layout.operator(StartPyMotionServerOperator.bl_idname, text="Start PyMotion Server")
    layout.operator(StopPyMotionServerOperator.bl_idname, text="Stop PyMotion Server")


def register():
    bpy.utils.register_class(StartPyMotionServerOperator)
    bpy.utils.register_class(StopPyMotionServerOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.utils.unregister_class(StartPyMotionServerOperator)
    bpy.utils.unregister_class(StopPyMotionServerOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()
    start_server()
