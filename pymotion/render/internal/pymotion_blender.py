# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import os
import socket
import struct
import sys
import time
import traceback
from threading import Thread

import bmesh
import bpy
import mathutils

# --- GLOBAL STATE ---
# We will store the server state in a dictionary to keep it tidy.
SERVER_STATE = {
    "host": "127.0.0.1",
    "port": 2222,
    "socket": None,
    "receive_thread": None,
    "finish_thread": False,
    "material_cache": {},
}


# --- NETWORK THREAD LOGIC ---
def receive_messages():
    """Listens for clients and schedules tasks for the main thread."""
    print("[PyMotion Thread] Receive thread started.")
    while not SERVER_STATE["finish_thread"]:
        try:
            print(
                f"[PyMotion Thread] Server listening on {SERVER_STATE['host']}:{SERVER_STATE['port']}..."
            )
            conn, addr = SERVER_STATE["socket"].accept()
            print(f"[PyMotion Thread] Accepted connection from {addr}")

            while not SERVER_STATE["finish_thread"]:
                # 1. Receive a complete command package
                command = receive_command(conn)
                if command is None:
                    print("[PyMotion Thread] Client disconnected or command invalid.")
                    conn.close()
                    break

                # print(f"[PyMotion Thread] Received command with ID: {command[0]}")

                # 2. Schedule this command to be run once on the main thread
                #    The `args` parameter passes our command tuple to the function.
                bpy.app.timers.register(dummy_callback, first_interval=0)
                bpy.app.timers.register(
                    lambda cmd=command: process_single_command(cmd), first_interval=0
                )

        except Exception as e:
            print(f"[PyMotion Thread] Error in accept loop: {e}")
            time.sleep(1)
    print("[PyMotion Thread] Receive thread finished.")


def dummy_callback():
    # This function does nothing, just wakes up the main thread
    # This is necessary because after a long time, Blender signals a KeyboardInterrupt and the first
    # function called will not be executed
    return None  # Do not repeat


def receive_command(conn):
    """Receives a full command (ID + data) from the client connection."""
    try:
        # --- Receive ID ---
        id_bytes = conn.recv(4)
        if not id_bytes:
            return None  # if empty byte object b'' is returned -> client closed the connection
        conn.sendall(struct.pack("<i", 1))  # ACK ID
        message_id = struct.unpack("<i", id_bytes)[0]

        # --- Receive Data (if any) ---
        data, colors, scales = (None, None, None)
        if not (
            message_id == 0 or message_id == 7
        ):  # clear_scene() nad other functions have no data
            size_data_bytes = conn.recv(4)
            size_data = struct.unpack("<i", size_data_bytes)[0]
            size_color_bytes = conn.recv(4)
            size_color = struct.unpack("<i", size_color_bytes)[0]
            size_scale_bytes = conn.recv(4)
            size_scale = struct.unpack("<i", size_scale_bytes)[0]
            data_type_indicator_bytes = conn.recv(4)
            data_type_indicator = struct.unpack("<i", data_type_indicator_bytes)[0]

            conn.sendall(struct.pack("<i", 1))  # ACK Metadata

            # Data
            if data_type_indicator == 0:  # string
                data_bytes = conn.recv(size_data)  # Receive string data
                data = data_bytes.decode("utf-8")
            elif data_type_indicator == 1:  # float array
                data_bytes = conn.recv(
                    4 * size_data
                )  # Receive float data (4 bytes per float)
                data = []
                for i in range(size_data):
                    data.append(struct.unpack("<f", data_bytes[i * 4 : i * 4 + 4])[0])
            else:
                raise ValueError(f"Unknown data type indicator: {data_type_indicator}")

            # Color
            if size_color > 0:
                color_bytes = conn.recv(
                    4 * size_color
                )  # Receive float data (4 bytes per float)
                colors = []
                for i in range(size_color):
                    colors.append(
                        struct.unpack("<f", color_bytes[i * 4 : i * 4 + 4])[0]
                    )
            else:
                colors = None

            # Scale
            if size_scale > 0:
                scales_bytes = conn.recv(
                    4 * size_scale
                )  # Receive float data (4 bytes per float)
                scales = []
                for i in range(size_scale):
                    scales.append(
                        struct.unpack("<f", scales_bytes[i * 4 : i * 4 + 4])[0]
                    )
            else:
                scales = None

        return (message_id, data, colors, scales)
    except Exception as e:
        print(f"[PyMotion Thread] Failed to receive command: {e}")
        return None


# --- MAIN THREAD TASK RUNNER ---
def process_single_command(command):
    """This function is executed by the main thread for EACH command."""
    try:
        message_id, data, color, scale = command
        # print(f"[PyMotion Main] Executing command ID: {message_id}")

        if message_id == 0:
            print("[PyMotion Main] Clearing scene")
            clear_scene()
        elif message_id == 1:
            print("[PyMotion Main] Rendering points")
            render_points(data, color, scale)
        elif message_id == 2:
            print("[PyMotion Main] Rendering orientations")
            render_orientations(data, scale)
        elif message_id == 3:
            print("[PyMotion Main] Rendering BVH")
            render_bvh(data, color, scale)
        elif message_id == 4:
            print("[PyMotion Main] Rendering checkerboard floor")
            render_checkerboard_floor(data)
        elif message_id == 5:
            print("[PyMotion Main] Rendering points timeline")
            render_points_timeline(data, color, scale)
        elif message_id == 6:
            print("[PyMotion Main] Setting up rendering")
            if data is None or len(data) < 4:
                setup_rendering()
            else:
                setup_rendering(
                    render_samples=int(round(data[0])),
                    viewport_samples=int(round(data[1])),
                    resolution_x=int(round(data[2])),
                    resolution_y=int(round(data[3])),
                )
        elif message_id == 7:
            print("[PyMotion Main] Rendering animation (bpy.ops.render.render(animation=True))")
            bpy.ops.render.render(animation=True)
        elif message_id == 8:
            print("[PyMotion Main] Setting camera position and focal length")
            try:
                cam_pos = data[:3]
                focal_len = data[3]
                set_camera(cam_pos, focal_len)
            except Exception as e:
                print(f"[PyMotion Main] Error setting camera: {e}")

        else:
            print(f"[PyMotion Main] Unknown message id {message_id}")

    except Exception:
        print(f"[PyMotion Main] CRITICAL ERROR while processing command {message_id}:")
        # Print the full, detailed error traceback to the console
        traceback.print_exc()

    # Return None so the timer only runs once and unregisters itself
    return None


# --- BLENDER MAIN THREAD LOGIC ---
# These functions modify Blender data and must run in the main thread.
# Render functions (clear_scene, render_points, etc.) go here.
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)


def setup_rendering(
    render_samples=32,
    viewport_samples=32,
    resolution_x=1920,
    resolution_y=1080,
):
    scene = bpy.context.scene

    # --- Output Settings ---
    # Set the container to MPEG-4 and the video codec to H.264 for a standard .mp4 file
    try:
        # Blender <= 4.x: use FFMPEG container settings
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
    except (TypeError, AttributeError):
        # Blender 5.0+: use media_type to switch to video output
        # No additional codec configuration needed - Blender 5.0+ handles it automatically
        scene.render.image_settings.media_type = "VIDEO"

    # --- Color Management Settings ---
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Very High Contrast"
    scene.view_settings.exposure = 1.8
    scene.view_settings.gamma = 0.5

    # --- Sampling Settings ---
    # Blender 4.x uses BLENDER_EEVEE_NEXT, Blender 5+ uses BLENDER_EEVEE
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    scene.eevee.taa_render_samples = int(render_samples)  # Render Samples
    scene.eevee.taa_samples = int(viewport_samples)  # Viewport Samples

    # --- Resolution Settings ---
    scene.render.resolution_x = int(resolution_x)
    scene.render.resolution_y = int(resolution_y)

    # --- World Setup ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.8, 0.68, 0.67, 1.0)
        bg_node.inputs["Strength"].default_value = 0.4

    # --- Sun Light Setup ---
    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN", align="WORLD", location=(0, 0, 0))
        sun_obj = bpy.context.active_object
        sun_obj.name = "Sun"
    sun_obj = bpy.data.objects["Sun"]

    sun_obj.data.color = (1.0, 0.88, 0.7)
    sun_obj.data.energy = 3.0
    sun_obj.data.angle = math.radians(5)  # Sun angle for soft shadows
    sun_obj.rotation_euler = (
        math.radians(25.399),
        math.radians(38.7803),
        math.radians(-0.660154),
    )


def set_camera(cam_pos, focal_len):
    """
    Sets the camera position, rotation, and focal length.
    cam_pos: (x, y, z) tuple or list
    focal_len: float (in mm)
    """
    scene = bpy.context.scene
    # Ensure camera exists
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(align="VIEW", location=(0, 0, 0), rotation=(0, 0, 0))
        camera_obj = bpy.context.active_object
        camera_obj.name = "Camera"
    camera_obj = bpy.data.objects["Camera"]
    # Set as active camera if not already
    if scene.camera is not camera_obj:
        scene.camera = camera_obj
    # Set camera position
    camera_obj.location = cam_pos
    camera_obj.rotation_mode = "XYZ"
    # Use the same rotation as in render_mocha_blender.py
    camera_obj.rotation_euler = (
        math.radians(54.1935),
        math.radians(-0.000027),
        math.radians(-0.230373),
    )
    camera_obj.data.lens = focal_len
    # Optionally print for debug
    print(f"[PyMotion Blender] Camera set: pos={cam_pos}, focal_length={focal_len}")


def render_points(data, color, scale):
    if data is None or color is None or scale is None:
        raise ValueError("Render points: Data, color or scale is None.")

    positions = []
    for i in range(0, len(data), 3):
        pos = data[i : i + 3]
        positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))

    colors = []
    for i in range(0, len(color), 3):
        col = color[i : i + 3]
        colors.append((col[0], col[1], col[2]))

    radius = []
    for i in range(0, len(scale), 1):
        rad = scale[i]
        radius.append(rad)

    mesh_data = bpy.data.meshes.new("SphereMesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=1)
    bm.to_mesh(mesh_data)
    bm.free()

    # Create a new collection for the points
    points_collection = bpy.data.collections.new("Points")
    bpy.context.scene.collection.children.link(points_collection)

    for i, pos in enumerate(positions):
        obj = bpy.data.objects.new(f"point_{i}", mesh_data)
        obj.location = pos
        obj.scale = (radius[i], radius[i], radius[i])
        points_collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.name = f"point_{i}"
        material = get_material(colors[i], no_illumination=True)
        if obj.data.materials:
            obj.data.materials[0] = material
            obj.data = obj.data.copy()
        else:
            obj.data.materials.append(material)


def render_points_timeline(data, color, scale):
    if data is None or color is None or scale is None:
        raise ValueError("Render points: Data, color or scale is None.")

    frames = int(round(data[0]))
    data_per_frame = (len(data) - 1) // frames

    positions = [[] for _ in range(frames)]
    for i in range(1, len(data), 3):
        current_frame = (i - 1) // data_per_frame
        pos = data[i : i + 3]
        positions[current_frame].append(mathutils.Vector([pos[0], pos[1], pos[2]]))

    colors = []
    for i in range(0, len(color), 3):
        col = color[i : i + 3]
        colors.append((col[0], col[1], col[2]))

    radius = []
    for i in range(0, len(scale), 1):
        rad = scale[i]
        radius.append(rad)

    # Initialize the points
    points = []
    mesh_data = bpy.data.meshes.new("SphereMesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=1)
    bm.to_mesh(mesh_data)
    bm.free()

    # Create a new collection for the points
    points_collection = bpy.data.collections.new("PointsTimeline")
    bpy.context.scene.collection.children.link(points_collection)

    for i, point in enumerate(positions[0]):
        obj = bpy.data.objects.new(f"point_t_{i}", mesh_data)
        obj.location = point
        obj.scale = (radius[i], radius[i], radius[i])
        points_collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.name = f"point_t_{i}"
        material = get_material(colors[i], no_illumination=True)
        if obj.data.materials:
            obj.data.materials[0] = material
            obj.data = obj.data.copy()
        else:
            obj.data.materials.append(material)
        points.append(obj)

    # Assign their position based on the frames
    for f in range(frames):
        for i, point in enumerate(positions[f]):
            points[i].location = point
            points[i].keyframe_insert(data_path="location", frame=f + 1)


def render_orientations(data, scale):
    if data is None or scale is None:
        raise ValueError("Render orientations: Data or scale is None.")

    quaternions = []
    positions = []
    for i in range(0, len(data), 7):  # 3 (positions) + 4 (quaternions)
        pos = data[i : i + 3]
        quat = data[i + 3 : i + 7]
        quaternions.append(mathutils.Quaternion([quat[0], quat[1], quat[2], quat[3]]))
        positions.append(mathutils.Vector([pos[0], pos[1], pos[2]]))

    scales = []
    for i in range(0, len(scale), 1):
        scale_i = scale[i]
        scales.append(scale_i)

    # Create a new collection
    orientations_collection = bpy.data.collections.new(name="Orientations")
    bpy.context.scene.collection.children.link(orientations_collection)

    for i, quat in enumerate(quaternions):
        bpy.ops.object.empty_add(type="ARROWS", location=positions[i])
        obj = bpy.context.object
        orientations_collection.objects.link(obj)
        obj.name = "orientation_{}".format(i)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat
        obj.scale = (scales[i], scales[i], scales[i])


def render_bvh(data, color, scale):
    if data is None or color is None or scale is None:
        raise ValueError("Render BVH: Data, color or scale is None.")

    data = data.split(".bvh")
    if len(data) != 2:
        raise ValueError("Invalid BVH data format.")
    bvh_path = data[0] + ".bvh"
    data = data[1].split(";")
    end_joints = data[:-2]
    axis_forward = data[-2]
    axis_up = data[-1]
    color = (
        color[0],
        color[1],
        color[2],
    )
    should_delete_file = scale[0] == 1

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


def render_checkerboard_floor(data):
    if data is None:
        raise ValueError("Render checkerboard floor: Data is None.")

    plane_size = data[0]
    checker_size = data[1]
    color1 = (
        data[2],
        data[3],
        data[4],
        1.0,
    )
    color2 = (
        data[5],
        data[6],
        data[7],
        1.0,
    )

    create_checkerboard_plane(plane_size, checker_size, color1, color2)


def get_material(color_rgb, no_illumination=False):  # Function to get or create material
    if color_rgb in SERVER_STATE["material_cache"]:
        return SERVER_STATE["material_cache"][color_rgb]  # Reuse existing material

    material = bpy.data.materials.new(name=f"PyMotionMat_{color_rgb}")  # Create new material
    material.use_nodes = True
    if no_illumination:
        background_node = material.node_tree.nodes.new(type="ShaderNodeBackground")
        background_node.inputs["Color"].default_value = (
            color_rgb[0],
            color_rgb[1],
            color_rgb[2],
            1.0,
        )  # RGBA (alpha 1.0)
        material.node_tree.links.new(
            material.node_tree.nodes["Material Output"].inputs["Surface"],
            background_node.outputs["Background"],
        )
    else:
        principled_bsdf = material.node_tree.nodes["Principled BSDF"]
        principled_bsdf.inputs["Base Color"].default_value = (
            color_rgb[0],
            color_rgb[1],
            color_rgb[2],
            1.0,
        )  # RGBA (alpha 1.0)

    SERVER_STATE["material_cache"][color_rgb] = material  # Store in cache
    return material


def generate_rig_representation(armature_obj, color, end_joints=None):
    if armature_obj.type != "ARMATURE":
        print("[PyMotion Blender] Selected object is not an armature!")
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
        if "twist" in bone.name:  # HARDCODED
            continue

        head_location = armature_obj.matrix_world @ bone.head_local
        tail_location = armature_obj.matrix_world @ bone.tail_local

        base_head_radius = 0.04
        base_cylinder_radius = 0.02

        distance_factor = (head_location - tail_location).length / 0.2
        distance_factor = min(1, math.exp(distance_factor) - 1)
        # TODO: ideally the distance_factor for the sphere is a mix between the previous bone and the current one

        sphere_head = create_sphere_at_location(
            head_location, base_head_radius * distance_factor, bone.name
        )
        sphere_head.data.materials.append(material)
        sphere_head.data = sphere_head.data.copy()
        sphere_current_collection = sphere_head.users_collection[0]
        # Link the new object to the collection
        rig_collection.objects.link(sphere_head)
        # Then, unlink the object from the main scene collection to avoid duplicates
        sphere_current_collection.objects.unlink(sphere_head)
        setup_constraints(sphere_head, bone.name, armature_obj)

        if end_joints is not None and bone.name in end_joints:
            continue

        cylinder = create_cylinder_between_points(
            bone,
            head_location,
            tail_location,
            base_cylinder_radius * distance_factor,
            bone.name,
        )
        cylinder.data.materials.append(material)
        cylinder.data = cylinder.data.copy()
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
    bmesh.ops.create_uvsphere(
        bm, u_segments=32, v_segments=16, radius=radius, calc_uvs=True
    )
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
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=5, radius1=radius, radius2=radius, depth=length
    )
    bm.to_mesh(mesh)
    bm.free()

    # Position it
    obj.location = (p1 + p2) / 2
    # Align local Z to direction
    up = mathutils.Vector((0, 0, 1))
    quat = up.rotation_difference(direction.normalized())
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat

    return obj


def setup_constraints(obj, target_bone_name, armature_object):
    constraint = obj.constraints.new(type="CHILD_OF")
    constraint.target = armature_object
    constraint.subtarget = target_bone_name


def create_checkerboard_plane(
    plane_size=2, checker_size=1, color1=(1, 1, 1, 1), color2=(0, 0, 0, 1)
):
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


# --- SERVER STARTUP AND REGISTRATION ---
def start_server():
    """Initializes the socket and starts the network thread."""
    if SERVER_STATE["receive_thread"] and SERVER_STATE["receive_thread"].is_alive():
        print("[PyMotion Main] Server is already running.")
        return

    try:
        port_str = sys.argv[sys.argv.index("--") + 1]
        SERVER_STATE["port"] = int(port_str)
    except (ValueError, IndexError):
        print(f"[PyMotion Main] Using default port {SERVER_STATE['port']}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((SERVER_STATE["host"], SERVER_STATE["port"]))
        sock.listen()
        SERVER_STATE["socket"] = sock
    except Exception as e:
        print(f"[PyMotion Main] FATAL: Could not bind to socket: {e}")
        return

    SERVER_STATE["finish_thread"] = False
    thread = Thread(target=receive_messages, daemon=True)
    thread.start()
    SERVER_STATE["receive_thread"] = thread
    print("[PyMotion Main] Server started successfully.")


# We don't need the full Operator class anymore, but we need a way to register the script
# This is now much simpler.
class PyMotionPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__


def register():
    start_server()


def unregister():
    print("[PyMotion Main] Unregistering and stopping server.")
    SERVER_STATE["finish_thread"] = True
    # Unblock the accept() call
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as dummy_socket:
            dummy_socket.connect((SERVER_STATE["host"], SERVER_STATE["port"]))
    except:
        pass
    if SERVER_STATE["socket"]:
        SERVER_STATE["socket"].close()


if __name__ == "__main__":
    # When Blender starts with -P, this block runs.
    start_server()
