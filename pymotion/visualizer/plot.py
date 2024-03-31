import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, ctx, Output, Input
import dash_bootstrap_components as dbc
import webbrowser
import os


class Viewer:
    def __init__(self, xy_size=2, z_size=2, use_reloader=False) -> None:
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

    def run(self):
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:8050")
        self.app.run_server(debug=True, use_reloader=self.use_reloader)

    def add_skeleton(
        self,
        data,
        parents,
        color="red",
        sphere_mode="scatter",
        scatter_size=3.0,
        line_width=2.0,
        radius_joints=0.025,
        resolution=np.pi / 8,
    ):
        assert data.ndim == 3 or data.ndim == 2, "'data' must have shape [frames, joints, 3] or [joints, 3]"
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
        self, center, sphere_mode="scatter", scatter_size=3.0, radius=1, color="red", resolution=np.pi / 8
    ):
        assert sphere_mode == "scatter" or sphere_mode == "mesh", "'sphere_mode' must be 'scatter' or 'mesh'"
        assert center.ndim == 2 or center.ndim == 1, "'center' must have shape [frames, 3] or [3]"
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

    def add_line(self, start, end, color="red", line_width=2.0):
        assert start.ndim == 2 or start.ndim == 1, "'start' must have shape [frames, 3] or [3]"
        assert end.ndim == 2 or end.ndim == 1, "'end' must have shape [frames, 3] or [3]"
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

    def add_floor(self, height=0, size=2, step=0.2, colors=["lightgrey", "darkgrey"]):
        self.static_objs.append(_create_floor(height=height, size=size, step=step, colors=colors))

    def _set_max_frames(self, value):
        self.max_frames = value
        self._update_layout()

    def _update_layout(self):
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

    def _set_up_callbacks(self):
        @callback(
            Output("graph-content", "figure"),
            Output("frame-input", "value"),
            Output("frames-slider", "value"),
            Input("frame-input", "value"),
            Input("frames-slider", "value"),
        )
        def update_frame(input_value, slider_value):
            id = ctx.triggered_id
            new_value = input_value if id == "frame-input" else slider_value
            return self._create_figure(new_value), new_value, new_value

    def _create_figure(self, frame=0):
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


def _create_mesh_sphere(center=np.array([0, 0, 0]), radius=1, color="red", resolution=np.pi / 8):
    """
    Creates a Plotly sphere mesh with customizable position, radius, and resolution.

    Args:
        center (tuple, optional): (x, y, z) coordinates of the sphere's center. Defaults to (0, 0, 0).
        radius (float, optional): Radius of the sphere. Defaults to 1.
        resolution (float, optional): Angular resolution for generating the sphere.
                                      Defaults to np.pi/32.

    Returns:
        plotly.graph_objects.Mesh3d: A Plotly Mesh3d object representing the sphere.
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


def _create_floor(height=0, size=1, step=0.1, colors=["lightgrey", "darkgrey"]):
    """
    Creates a floor with a checkerboard pattern.

    Args:
        height (float, optional): Height (z-coordinate) of the floor. Defaults to 0.
        size (float, optional): Size of the floor along x and y directions. Defaults to 1.
        colors (list, optional):  List of two colors for the checkerboard pattern. Defaults to ['lightgrey', 'darkgrey'].

    Returns:
        plotly.graph_objects.Mesh3d: Plotly Mesh3d object representing the floor.
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
    joints,
    parents,
    color="red",
    sphere_mode="scatter",
    scatter_size=3.0,
    line_width=2.0,
    radius_joints=0.025,
    resolution=np.pi / 8,
):
    """
    Creates a Plotly mesh for a skeleton using a NumPy array of joint positions.

    Args:
        data (np.array): A NumPy array of shape (num_joints, 3) containing the x, y, and z coordinates of the joints.
        color (str, optional): Color of the skeleton. Defaults to "red".
        resolution (float, optional): Angular resolution for generating the spheres. Defaults to np.pi/32.

    Returns:
        list: A list of Plotly go.Mesh3d objects representing the spheres.
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
