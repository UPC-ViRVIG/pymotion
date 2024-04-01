from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, ctx, Output, Input
import dash_bootstrap_components as dbc
import webbrowser
import os


class Viewer:
    """
    Class to represent a 3D motion visualization tool using Plotly and Dash.
    """

    def __init__(self, xy_size: float = 2, z_size: float = 2, use_reloader: bool = False) -> None:
        """
        Initializes the Viewer.

        Args:
            xy_size (float): Size of the viewer in the X and Y dimensions. Defaults to 2.
            z_size (float): Size of the viewer in the Z dimension. Defaults to 2.
            use_reloader (bool): Whether to use automatic reloading for development. Defaults to False.
        """
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.static_objs = []  # Array of Plotly Objects
        self.dynamic_data = {
            "skeleton": [],
            "sphere": [],
            "line": [],
        }  # Keys: Type, Value: dict with parameters + NumPy Array [frames, ...]
        self.xy_size = xy_size
        self.z_size = z_size
        self.use_reloader = use_reloader
        self._set_max_frames(0)
        self._set_up_callbacks()

    def run(self) -> None:
        """
        Runs the Dash application to start the viewer.
        """

        # Check for debugging environment to avoid duplicated browser tabs
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:8050")
        self.app.run_server(debug=True, use_reloader=self.use_reloader)

    def add_skeleton(
        self,
        data: np.ndarray,
        parents: np.ndarray,
        color: str = "red",
        sphere_mode: str = "scatter",
        scatter_size: float = 3.0,
        line_width: float = 2.0,
        radius_joints: float = 0.025,
        resolution: float = np.pi / 8,
    ) -> None:
        """
        Adds a skeleton to the viewer.

        Args:
            data (np.ndarray): NumPy array of shape [frames, joints, 3] or [joints, 3] containing joint positions.
            parents (np.ndarray): NumPy array indicating the parent joint for each joint, enabling line connections.
            color (str): Color of the skeleton elements. Defaults to "red".
            sphere_mode (str): 'scatter' for a point cloud representation, or 'mesh' for a 3D mesh sphere. Defaults to 'scatter'.
            scatter_size (float): Size of scatter markers if 'sphere_mode' is 'scatter'. Defaults to 3.0.
            line_width (float): Width of the lines connecting skeleton joints. Defaults to 2.0.
            radius_joints (float): Radius of joints if represented as spheres ('mesh' mode). Defaults to 0.025.
            resolution (float): Angular resolution for sphere meshes (`mesh` mode). Defaults to np.pi/8.
        """

        assert data.ndim in (2, 3), "'data' must have shape [frames, joints, 3] or [joints, 3]"

        if data.ndim == 2 or data.shape[0] == 1:
            self.static_objs.extend(
                _create_skeleton(
                    data if data.ndim == 2 else data[0],
                    parents,
                    color=color,
                    sphere_mode=sphere_mode,
                    scatter_size=scatter_size,
                    line_width=line_width,
                    radius_joints=radius_joints,
                    resolution=resolution,
                )
            )
        else:
            self.dynamic_data["skeleton"].append(
                {
                    "data": data,
                    "parents": parents,
                    "color": color,
                    "sphere_mode": sphere_mode,
                    "scatter_size": scatter_size,
                    "line_width": line_width,
                    "radius_joints": radius_joints,
                    "resolution": resolution,
                }
            )
            self._set_max_frames(max(self.max_frames, data.shape[0]))

    def add_sphere(
        self,
        center: np.ndarray,
        sphere_mode: str = "scatter",
        scatter_size: float = 3.0,
        radius: float = 1,
        color: str = "red",
        resolution: float = np.pi / 8,
    ) -> None:
        """
        Adds a sphere to the viewer.

        Args:
            center (np.ndarray): NumPy array of shape [frames, 3] or [3] specifying the sphere's center coordinates.
            sphere_mode (str): 'scatter' for a point cloud representation, or 'mesh' for a 3D mesh sphere. Defaults to 'scatter'.
            scatter_size (float): Size of points if 'sphere_mode' is 'scatter'. Defaults to 3.0.
            radius (float): Radius of the sphere if 'sphere_mode' is 'mesh'. Defaults to 1.
            color (str): Color of the sphere. Defaults to 'red'.
            resolution (float): Angular resolution for creating the sphere mesh in 'mesh' mode. Defaults to np.pi/8.
        """

        assert sphere_mode in ("scatter", "mesh"), "'sphere_mode' must be 'scatter' or 'mesh'"
        assert center.ndim in (1, 2), "'center' must have shape [frames, 3] or [3]"

        if center.ndim == 1 or center.shape[0] == 1:
            if sphere_mode == "scatter":
                self.static_objs.append(
                    go.Scatter3d(
                        x=[center[0]],
                        y=[center[1]],
                        z=[center[2]],
                        mode="markers",
                        marker=dict(color=color, size=scatter_size),
                    )
                )
            elif sphere_mode == "mesh":
                self.static_objs.append(_create_mesh_sphere(center, radius, color, resolution))
        else:
            self.dynamic_data["sphere"].append(
                {
                    "center": center,
                    "sphere_mode": sphere_mode,
                    "scatter_size": scatter_size,
                    "radius": radius,
                    "color": color,
                    "resolution": resolution,
                }
            )
            self._set_max_frames(max(self.max_frames, center.shape[0]))

    def add_line(
        self, start: np.ndarray, end: np.ndarray, color: str = "red", line_width: float = 2.0
    ) -> None:
        """
        Adds a line to the viewer.

        Args:
            start (np.ndarray): NumPy array of shape [frames, 3] or [3] for the starting coordinates of the line.
            end (np.ndarray): NumPy array of shape [frames, 3] or [3] for the ending coordinates of the line.
            color (str): Color of the line. Defaults to 'red'.
            line_width (float): Width of the line. Defaults to 2.0.
        """

        assert start.ndim in (1, 2), "'start' must have shape [frames, 3] or [3]"
        assert end.ndim in (1, 2), "'end' must have shape [frames, 3] or [3]"
        assert start.ndim == end.ndim, "'start' and 'end' must have the same number of dimensions"

        if start.ndim == 1 or start.shape[0] == 1:
            self.static_objs.append(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=line_width),
                )
            )
        else:
            self.dynamic_data["line"].append(
                {
                    "start": start,
                    "end": end,
                    "color": color,
                    "line_width": line_width,
                }
            )
            self._set_max_frames(max(self.max_frames, start.shape[0]))

    def add_floor(
        self, height: float = 0, size: float = 2, step: float = 0.2, colors: list = ["lightgrey", "darkgrey"]
    ) -> None:
        """
        Adds a checkered floor plane to the viewer.

        Args:
            height (float): Height (z-coordinate) of the floor plane. Defaults to 0.
            size (float): Size of the floor in the x and y directions. Defaults to 2.
            step (float): Size of each checkerboard square. Defaults to 0.2.
            colors (list): A list of two colors for the checkerboard pattern. Defaults to ["lightgrey", "darkgrey"].
        """
        self.static_objs.append(_create_floor(height=height, size=size, step=step, colors=colors))

    def _set_max_frames(self, value: int) -> None:
        """
        Updates the 'max_frames' attribute, which is used to track the maximum number of frames across all dynamic data.

        Args:
            value (int): The new value to set for 'max_frames'.
        """
        self.max_frames = value
        self._update_layout()

    def _update_layout(self) -> None:
        """
        Updates the layout of the Dash application. This includes setting the plot size and the slider range.
        """
        self.app.layout = html.Div(
            [
                html.H1(children="PyMotion Viewer", style={"textAlign": "center"}),
                dcc.Graph(
                    id="graph-content",
                    style={"height": "80vh", "margin": "auto"},
                    responsive=True,
                    config={"displayModeBar": False},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                type="number",
                                value=0,
                                style={"textAlign": "center"},
                                id="frame-input",
                            ),
                            width=1,
                        ),
                        dbc.Col(
                            dcc.Slider(
                                min=0,
                                max=self.max_frames - 1,
                                step=1,
                                value=0,
                                marks=None,
                                id="frames-slider",
                            ),
                            width=8,
                        ),
                    ],
                    justify="center",
                ),
            ]
        )

    def _set_up_callbacks(self) -> None:
        """
        Sets up the Dash callbacks to enable interactivity (updating frame based on input or slider interaction).
        """

        @callback(
            Output("graph-content", "figure"),
            Output("frame-input", "value"),
            Output("frames-slider", "value"),
            Input("frame-input", "value"),
            Input("frames-slider", "value"),
        )
        def update_frame(input_value, slider_value):
            """Handles changes to the frame input or slider."""
            id = ctx.triggered_id
            new_value = input_value if id == "frame-input" else slider_value
            return self._create_figure(new_value), new_value, new_value

    def _create_figure(self, frame: int = 0) -> go.Figure:
        """
        Creates the Plotly figure for a given frame of the visualization.

        Args:
            frame (int): The frame number to display. Defaults to 0.

        Returns:
            go.Figure: The Plotly Figure object representing the scene at the specified frame.
        """
        data = []
        for key, value in self.dynamic_data.items():
            if key == "skeleton":
                for skeleton in value:
                    if frame < len(skeleton["data"]):
                        data.extend(
                            _create_skeleton(
                                joints=skeleton["data"][frame],
                                parents=skeleton["parents"],
                                color=skeleton["color"],
                                sphere_mode=skeleton["sphere_mode"],
                                scatter_size=skeleton["scatter_size"],
                                line_width=skeleton["line_width"],
                                radius_joints=skeleton["radius_joints"],
                                resolution=skeleton["resolution"],
                            )
                        )
            elif key == "sphere":
                for sphere in value:
                    if frame < len(sphere["center"]):
                        if sphere["sphere_mode"] == "scatter":
                            data.append(
                                go.Scatter3d(
                                    x=[sphere["center"][frame][0]],
                                    y=[sphere["center"][frame][1]],
                                    z=[sphere["center"][frame][2]],
                                    mode="markers",
                                    marker=dict(color=sphere["color"], size=sphere["scatter_size"]),
                                )
                            )
                        elif sphere["sphere_mode"] == "mesh":
                            data.append(
                                _create_mesh_sphere(
                                    center=sphere["center"][frame],
                                    radius=sphere["radius"],
                                    color=sphere["color"],
                                    resolution=sphere["resolution"],
                                )
                            )
            elif key == "line":
                for line in value:
                    if frame < len(line["start"]):
                        data.append(
                            go.Scatter3d(
                                x=[line["start"][frame][0], line["end"][frame][0]],
                                y=[line["start"][frame][1], line["end"][frame][1]],
                                z=[line["start"][frame][2], line["end"][frame][2]],
                                mode="lines",
                                line=dict(color=line["color"], width=line["line_width"]),
                            )
                        )
        fig = go.Figure(self.static_objs + data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-self.xy_size, self.xy_size]),
                yaxis=dict(range=[-self.xy_size, self.xy_size]),
                zaxis=dict(range=[-self.z_size, self.z_size]),
            ),
            scene_aspectmode="manual",
            scene_aspectratio=dict(x=1, y=1, z=self.z_size / self.xy_size),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
        )
        return fig


def _create_mesh_sphere(
    center: np.ndarray = np.array([0, 0, 0]),
    radius: float = 1,
    color: str = "red",
    resolution: float = np.pi / 8,
) -> go.Mesh3d:
    """
    Creates a 3D mesh representation of a sphere with customizable position, radius, and resolution.

    Args:
        center (np.ndarray): (x, y, z) coordinates of the sphere's center. Defaults to np.array[(0, 0, 0)].
        radius (float): Radius of the sphere. Defaults to 1.
        color (str): Color of the sphere. Defaults to 'red'.
        resolution (float): Angular resolution for generating the sphere. Defaults to np.pi/8.

    Returns:
        go.Mesh3d: A Plotly Mesh3d object representing the sphere.
    """
    d = resolution  # Angular spacing

    theta, phi = np.mgrid[0 : np.pi + d : d, 0 : 2 * np.pi : d]
    x = np.sin(theta) * np.cos(phi)  # Adjust for center
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Adjust radius
    x *= radius
    y *= radius
    z *= radius

    # Displace sphere to center
    x += center[0]
    y += center[1]
    z += center[2]

    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    x, y, z = points

    return go.Mesh3d(x=x, y=y, z=z, color=color, opacity=1.00, alphahull=0)


def _create_floor(
    height: float = 0.0, size: float = 1.0, step: float = 0.1, colors: list[str] = ["lightgrey", "darkgrey"]
) -> go.Mesh3d:
    """
    Creates a floor plane with a checkerboard pattern.

    Args:
        height (float): Height (z-coordinate) of the floor. Defaults to 0.0.
        size (float): Size of the floor along x and y directions. Defaults to 1.0.
        step (float): Size of each checkerboard square. Defaults to 0.1.
        colors (list[str]): A list of two colors for the checkerboard pattern. Defaults to ["lightgrey", "darkgrey"].

    Returns:
        go.Mesh3d: Plotly Mesh3d object representing the floor.
    """

    x = []
    y = []
    z = []
    i = []
    j = []
    k = []
    vertexcolor = []

    # Create a checkerboard pattern
    for i_x, v_x in enumerate(np.arange(-size, size + step, step)):
        for i_j, v_j in enumerate(np.arange(-size, size + step, step)):
            a_i = len(x)  # absolute index
            x += [v_x, v_x + step, v_x + step, v_x]
            y += [v_j, v_j, v_j + step, v_j + step]
            z += [height, height, height, height]
            c_i = 0 if (i_x + i_j) % 2 == 0 else 1
            vertexcolor += [colors[c_i], colors[c_i], colors[c_i], colors[c_i]]
            i += [a_i, a_i + 3]
            j += [a_i + 1, a_i + 1]
            k += [a_i + 3, a_i + 2]

    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=1.0, delaunayaxis="z", vertexcolor=vertexcolor)


def _create_skeleton(
    joints: np.ndarray,
    parents: np.ndarray,
    color: str = "red",
    sphere_mode: str = "scatter",
    scatter_size: float = 3.0,
    line_width: float = 2.0,
    radius_joints: float = 0.025,
    resolution: float = np.pi / 8,
) -> list[go.Scatter3d | go.Mesh3d]:
    """
    Creates Plotly objects for a skeleton using a NumPy array of joint positions.

    Args:
        joints (np.ndarray): NumPy array of shape (num_joints, 3) containing the x, y, and z coordinates of the joints.
        parents (np.ndarray): NumPy array defining the parent joint for each joint, enabling line connections.
        color (str): Color of the skeleton elements. Defaults to 'red'.
        sphere_mode (str): Controls sphere representation ('scatter' or 'mesh'). Defaults to 'scatter'.
        scatter_size (float): Size of scatter markers if 'sphere_mode' is 'scatter'. Defaults to 3.0.
        line_width (float): Width of the lines connecting skeleton joints. Defaults to 2.0.
        radius_joints (float): Radius of joints if represented as spheres ('mesh' mode). Defaults to 0.025.
        resolution (float): Angular resolution for sphere meshes (`mesh` mode). Defaults to np.pi/8.

    Returns:
        list[go.Scatter3d | go.Mesh3d]: A list of Plotly go.Scatter3d and/or go.Mesh3d objects representing the skeleton.
    """

    sphere_data = []
    line_data = []
    for joint_idx, parent_idx in enumerate(parents):
        joint = joints[joint_idx]
        if sphere_mode == "scatter":
            sphere_data.append(
                go.Scatter3d(
                    x=[joint[0]],
                    y=[joint[1]],
                    z=[joint[2]],
                    mode="markers",
                    marker=dict(color=color, size=scatter_size),
                )
            )
        else:
            sphere_data.append(
                _create_mesh_sphere(center=joint, radius=radius_joints, color=color, resolution=resolution)
            )
        if parent_idx is not None and parent_idx >= 0:
            parent = joints[parent_idx]
            line_data.append(
                go.Scatter3d(
                    x=[joint[0], parent[0]],
                    y=[joint[1], parent[1]],
                    z=[joint[2], parent[2]],
                    mode="lines",
                    line=dict(color=color, width=line_width),
                )
            )

    return sphere_data + line_data
