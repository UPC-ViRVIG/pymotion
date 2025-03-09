import importlib

REQUIRED_PACKAGES = ["plotly", "dash", "dash_bootstrap_components"]


def _check_dependencies():
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        message = (
            f"The 'pymotion.render.viewer' subpackage requires the following packages to be installed: "
            f"{', '.join(missing_packages)} \n"
            f"Please install them using pip: \n"
            f"pip install upc-pymotion[viewer] \n"
        )
        raise ImportError(message)


# Call the check when the subpackage is imported
_check_dependencies()


from __future__ import annotations
import json
from typing import List, Union
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, clientside_callback
import dash_bootstrap_components as dbc
import webbrowser
import os


class Viewer:
    """
    Class to represent a 3D motion visualization tool using Plotly and Dash.
    """

    def __init__(
        self,
        xy_size: Union[float, List[float]] = 2,
        z_size: Union[float, List[float]] = 2,
        framerate: int = 60,
        use_reloader: bool = False,
    ) -> None:
        """
        Initializes the Viewer.

        Args:
            xy_size (float or list(float)): Size of the viewer in the X and Y dimensions. Defaults to 2. Can be interval [-3, 2].
            z_size (float or list(float)): Size of the viewer in the Z dimension. Defaults to 2. Can be interval [-3, 2].
            framerate (int): Frames per second of the visualization. Defaults to 60 fps.
            use_reloader (bool): Whether to use automatic reloading for development. Defaults to False.
        """
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "PyMotion Viewer"
        self.static_objs = []  # Array of Plotly Objects
        self.dynamic_data = {
            "skeleton": [],
            "sphere": [],
            "line": [],
        }  # Keys: Type, Value: dict with parameters and a "data" NumPy Array [frames, ...]
        self.playing = False  # Flag to indicate if the animation is playing
        self.frametime = 1000 / framerate  # Time between frames in milliseconds
        self.use_reloader = use_reloader

        if isinstance(xy_size, list):
            assert len(xy_size) == 2, "if you pass interval, it should contain only 2 values"
            assert xy_size[0] < xy_size[1], "if you pass interval, first value should be less than second"
            xy_dict = dict(range=[xy_size[0], xy_size[1]])
        else:
            xy_dict = dict(range=[-xy_size, xy_size])
        if isinstance(z_size, list):
            assert len(z_size) == 2, "if you pass interval, it should contain only 2 values"
            assert z_size[0] < z_size[1], "if you pass interval, first value should be less than second"
            z_dict = dict(range=[z_size[0], z_size[1]])
        else:
            z_dict = dict(range=[-z_size, z_size])

        aspect_ratio_value = (z_dict["range"][1] - z_dict["range"][0]) / (
            xy_dict["range"][1] - xy_dict["range"][0]
        )

        # Create a default Figure
        self.fig = go.Figure()
        self.fig.update_layout(
            scene=dict(
                xaxis=xy_dict,
                yaxis=xy_dict,
                zaxis=z_dict,
            ),
            scene_aspectmode="manual",
            scene_aspectratio=dict(x=1, y=1, z=aspect_ratio_value),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
            uirevision="constant",  # avoid resetting the camera view
        )

        self._set_max_frames(0)

    def run(self) -> None:
        """
        Runs the Dash application to start the viewer.
        """
        # Create objects
        figure = self._create_figure()

        # Store changes for all frames
        frames = [self._update_figure(frame) for frame in range(self.max_frames)]

        # Create Dash layout
        self._update_layout(figure, frames)

        # Set up callbacks
        self._set_up_callbacks()

        # Check for debugging environment to avoid duplicated browser tabs
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:8050")
        self.app.run_server(debug=True, use_reloader=self.use_reloader)

    def add_skeleton(
        self,
        data: np.ndarray,
        parents: np.ndarray,
        color: str = "red",
        size: float = 3.0,
        line_width: float = 2.0,
    ) -> None:
        """
        Adds a skeleton to the viewer.

        Args:
            data (np.ndarray): NumPy array of shape [frames, joints, 3] or [joints, 3] containing joint positions.
            parents (np.ndarray): NumPy array indicating the parent joint for each joint, enabling line connections.
            color (str): Color of the skeleton elements. Defaults to "red".
            size (float): Size of scatter markers. Defaults to 3.0.
            line_width (float): Width of the lines connecting skeleton joints. Defaults to 2.0.
        """

        assert data.ndim in (2, 3), "'data' must have shape [frames, joints, 3] or [joints, 3]"
        assert parents.ndim == 1, "'parents' must have shape [joints]"
        assert data.shape[-1] == 3, "'data' must have shape [..., 3]"
        assert data.shape[-2] == parents.shape[0], "'data' and 'parents' must have the same number of joints"

        if data.dtype != np.float64:
            data = data.astype(np.float64)
        if parents.dtype != np.int64:
            parents = parents.astype(np.int64)

        if data.ndim == 2 or data.shape[0] == 1:
            self.static_objs.extend(
                _create_skeleton(
                    data if data.ndim == 2 else data[0],
                    parents,
                    color=color,
                    size=size,
                    line_width=line_width,
                )
            )
        else:
            self.dynamic_data["skeleton"].append(
                {
                    "data": data,
                    "parents": parents,
                    "color": color,
                    "size": size,
                    "line_width": line_width,
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
        Note: when sphere_mode os "mesh", it can be slow for large number of frames and/or spheres.

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
        assert center.shape[-1] == 3, "'center' must have shape [..., 3]"

        if center.dtype != np.float64:
            center = center.astype(np.float64)

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
        assert start.shape[-1] == 3 and end.shape[-1], "'start' and 'end' must have shape [..., 3]"
        assert start.shape[0] == end.shape[0], "'start' and 'end' must have the same number of frames"

        if start.dtype != np.float64:
            start = start.astype(np.float64)
        if end.dtype != np.float64:
            end = end.astype(np.float64)

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

    def _update_layout(self, figure: go.Figure, frames: list) -> None:
        """
        Updates the layout of the Dash application. This includes setting the plot size and the slider range.
        """
        frames_json = json.dumps(frames)
        self.app.layout = html.Div(
            [
                dcc.Graph(
                    id="graph-content",
                    figure=figure,
                    style={"height": "80vh", "margin": "auto"},
                    responsive=True,
                    config={
                        "displayModeBar": False,
                        "scrollZoom": False,
                    },  # TODO: for now disable scroll zoom until plotly solves the bug: scroll zoom does not save the internal state for uirevision https://github.com/plotly/plotly.js/issues/5004
                ),
                dcc.Store(id="store-frames", data=frames_json),
                dcc.Store(id="store-playback-speed", data=1),
                dcc.Interval(
                    id="interval-timer",
                    interval=self.frametime,
                    disabled=True,
                    n_intervals=0,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            id="play-stop-button",
                                            n_clicks=0,
                                            children="Play",
                                            color="primary",
                                            style={
                                                "width": "90%",
                                                "margin-left": "5%",
                                            },
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            id="playback-speed",
                                            n_clicks=0,
                                            children="1x",
                                            color="light",
                                            style={"width": "90%"},
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            type="number",
                                            value=0,
                                            style={"textAlign": "center", "width": "90%"},
                                            id="frame-input",
                                        ),
                                        width=7,
                                    ),
                                ],
                                className="g-1",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            html.Div(
                                dcc.Slider(
                                    min=0,
                                    max=self.max_frames - 1,
                                    step=1,
                                    value=0,
                                    marks=None,
                                    id="frames-slider",
                                ),
                                style={
                                    "width": "100%",
                                    "height": "100%",
                                    "margin-left": "-25px",
                                    "margin-bottom": "-25px",
                                },
                            ),
                            width=8,
                        ),
                    ],
                    align="center",
                ),
            ]
        )

    def _set_up_callbacks(self) -> None:
        """
        Sets up the Dash callbacks to enable interactivity (updating frame based on input or slider interaction).
        """

        clientside_callback(
            """
            function updateFrame(input_value, slider_value, n_intervals, store_frames, playback_speed) {
                // Get the triggered component ID
                let triggeredId = 'none';
                if (dash_clientside.callback_context.triggered[0] != undefined) 
                {
                    triggeredId = dash_clientside.callback_context.triggered[0].prop_id.split('.')[0];
                }

                // Parse the precomputed figures from the store only once
                if (!window.precomputedFigures) window.precomputedFigures = JSON.parse(store_frames);
                
                let frame;
                if (triggeredId === 'frame-input') {
                    frame = input_value;
                } else if (triggeredId === 'frames-slider') {
                    frame = slider_value;
                } else if (triggeredId === 'interval-timer') {
                    // Handle autoplay logic if needed
                    frame = (input_value + playback_speed) % window.precomputedFigures.length;
                } else {
                    frame = 0; // Initial frame or fallback
                }

                const data = window.precomputedFigures[frame];
                
                // Update the graph using extendData
                return [[data[0], data[1], data[2]], frame, frame];
            }
            """,
            Output("graph-content", "extendData"),
            Output("frame-input", "value"),
            Output("frames-slider", "value"),
            Input("frame-input", "value"),
            Input("frames-slider", "value"),
            Input("interval-timer", "n_intervals"),
            Input("store-frames", "data"),
            State("store-playback-speed", "data"),
        )

        @callback(
            Output("play-stop-button", "children"),
            Output("play-stop-button", "color"),
            Output("interval-timer", "disabled"),
            Input("play-stop-button", "n_clicks"),
        )
        def update_button_text(play_stop_clicks):
            """Updates the button text based on play/stop status."""
            if play_stop_clicks % 2 == 1:
                return "Stop", "danger", False
            else:
                return "Play", "primary", True

        @callback(
            Output("playback-speed", "children"),
            Output("store-playback-speed", "data"),
            Input("playback-speed", "n_clicks"),
        )
        def update_playback_speed(playback_speed_clicks):
            """Updates the playback speed based on button clicks."""
            speed = 1
            op = playback_speed_clicks % 4
            if op == 1:
                speed = 2
            elif op == 2:
                speed = 4
            elif op == 3:
                speed = 8
            return f"{speed}x", speed

    def _create_figure(self, frame: int = 0) -> go.Figure:
        """
        Creates the Plotly figure for a given frame of the visualization.

        Args:
            frame (int): The frame number to display. Defaults to 0.

        Returns:
            go.Figure: A Plotly Figure object containing the visualization.
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
                                size=skeleton["size"],
                                line_width=skeleton["line_width"],
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
        self.fig.data = []
        self.fig.add_traces(data + self.static_objs)
        return self.fig

    def _update_figure(self, frame: int = 0):
        """
        Creates the Plotly figure for a given frame of the visualization.

        Args:
            frame (int): The frame number to display. Defaults to 0.

        Returns:
            Patch: A Patch object containing the updated Plotly figure.
        """
        data = {"x": [], "y": [], "z": []}
        indices = []
        number_of_objects = []
        i = 0
        for key, value in self.dynamic_data.items():
            if key == "skeleton":
                for skeleton in value:
                    if frame < len(skeleton["data"]):
                        i = _update_skeleton(
                            data=data,
                            indices=indices,
                            number_of_objects=number_of_objects,
                            i=i,
                            joints=skeleton["data"][frame],
                            parents=skeleton["parents"],
                        )
            elif key == "sphere":
                for sphere in value:
                    if frame < len(sphere["center"]):
                        if sphere["sphere_mode"] == "scatter":
                            indices.append(i)
                            data["x"].append([sphere["center"][frame][0]])
                            data["y"].append([sphere["center"][frame][1]])
                            data["z"].append([sphere["center"][frame][2]])
                            number_of_objects.append(1)
                        elif sphere["sphere_mode"] == "mesh":
                            x, y, z = _get_mesh_sphere_position(
                                center=sphere["center"][frame],
                                radius=sphere["radius"],
                                resolution=sphere["resolution"],
                            )
                            indices.append(i)
                            data["x"].append(x.tolist())
                            data["y"].append(y.tolist())
                            data["z"].append(z.tolist())
                            number_of_objects.append(len(data["x"][-1]))
                    i += 1
            elif key == "line":
                for line in value:
                    if frame < len(line["start"]):
                        indices.append(i)
                        data["x"].append([line["start"][frame][0], line["end"][frame][0]])
                        data["y"].append([line["start"][frame][1], line["end"][frame][1]])
                        data["z"].append([line["start"][frame][2], line["end"][frame][2]])
                        number_of_objects.append(2)
                    i += 1
        number_of_objects_per_axis = {"x": number_of_objects, "y": number_of_objects, "z": number_of_objects}
        return (data, indices, number_of_objects_per_axis)


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
    x, y, z = _get_mesh_sphere_position(center=center, radius=radius, resolution=resolution)
    return go.Mesh3d(x=x, y=y, z=z, color=color, opacity=1.00, alphahull=0)


def _get_mesh_sphere_position(
    center: np.ndarray = np.array([0, 0, 0]), radius: float = 1, resolution: float = np.pi / 8
) -> np.ndarray:
    """
    Gets the points of a 3D mesh representation of a sphere with customizable position, radius, and resolution.

    Args:
        center (np.ndarray): (x, y, z) coordinates of the sphere's center. Defaults to np.array[(0, 0, 0)].
        radius (float): Radius of the sphere. Defaults to 1.
        resolution (float): Angular resolution for generating the sphere. Defaults to np.pi/8.

    Returns:
        np.ndarray: A NumPy array of shape (n_points,) containing the x coordinate of the sphere.
        np.ndarray: A NumPy array of shape (n_points,) containing the y coordinate of the sphere.
        np.ndarray: A NumPy array of shape (n_points,) containing the z coordinate of the sphere.
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
    return x, y, z


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
    size: float = 3.0,
    line_width: float = 2.0,
) -> list[go.Scatter3d]:
    """
    Creates Plotly objects for a skeleton using a NumPy array of joint positions.

    Args:
        joints (np.ndarray): NumPy array of shape (num_joints, 3) containing the x, y, and z coordinates of the joints.
        parents (np.ndarray): NumPy array defining the parent joint for each joint, enabling line connections.
        color (str): Color of the skeleton elements. Defaults to 'red'.
        size (float): Size of scatter markers. Defaults to 3.0.
        line_width (float): Width of the lines connecting skeleton joints. Defaults to 2.0.

    Returns:
        list[go.Scatter3d]: A list of Plotly go.Scatter3d objects representing the skeleton.
    """

    sphere_data = []
    line_data = []
    for joint_idx, parent_idx in enumerate(parents):
        joint = joints[joint_idx]
        sphere_data.append(
            go.Scatter3d(
                x=[joint[0]],
                y=[joint[1]],
                z=[joint[2]],
                mode="markers",
                marker=dict(color=color, size=size),
            )
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


def _update_skeleton(
    data: dict,
    indices: list,
    number_of_objects: list,
    i: int,
    joints: np.ndarray,
    parents: np.ndarray,
) -> int:
    for joint_idx, parent_idx in enumerate(parents):
        joint = joints[joint_idx]
        indices.append(i)
        data["x"].append([joint[0]])
        data["y"].append([joint[1]])
        data["z"].append([joint[2]])
        number_of_objects.append(1)
        i += 1
    for joint_idx, parent_idx in enumerate(parents):
        joint = joints[joint_idx]
        if parent_idx is not None and parent_idx >= 0:
            parent = joints[parent_idx]
            indices.append(i)
            number_of_objects.append(2)
            data["x"].append([joint[0], parent[0]])
            data["y"].append([joint[1], parent[1]])
            data["z"].append([joint[2], parent[2]])
            i += 1
    return i
