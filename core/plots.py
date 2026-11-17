"""Plotly figure builders."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from matplotlib.tri import Triangulation, LinearTriInterpolator

from .mesh import Mesh, REG_CASING, REG_FILLER

FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _circle_shapes(mesh: Mesh, scale=1e3, color="rgba(255,255,255,0.55)"):
    shapes = []
    r = mesh.radius * scale
    for x, y in mesh.centres * scale:
        shapes.append(dict(type="circle", xref="x", yref="y", x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                           line=dict(color=color, width=1)))
    return shapes


def field_figure(mesh: Mesh, T: np.ndarray, title: str, zmin=None, zmax=None, nx=700):
    tri = Triangulation(mesh.p[:, 0], mesh.p[:, 1], mesh.t)
    interp = LinearTriInterpolator(tri, T)
    ny = max(40, int(nx * mesh.H / mesh.W))
    xs = np.linspace(0, mesh.W, nx)
    ys = np.linspace(0, mesh.H, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.ma.filled(interp(X, Y), np.nan)
    fig = go.Figure(go.Heatmap(
        x=xs * 1e3, y=ys * 1e3, z=Z, colorscale="Inferno", zmin=zmin, zmax=zmax, zsmooth="best",
        colorbar=dict(title="T (°C)"),
        hovertemplate="x = %{x:.1f} mm<br>y = %{y:.1f} mm<br>T = %{z:.2f} °C<extra></extra>"))
    fig.add_trace(go.Contour(x=xs * 1e3, y=ys * 1e3, z=Z, showscale=False, contours_coloring="none",
                             line=dict(color="rgba(255,255,255,0.35)", width=0.8), ncontours=14,
                             hoverinfo="skip"))
    fig.update_layout(shapes=_circle_shapes(mesh), title=title, font=FONT,
                      margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(title="x (mm), along vehicle", scaleanchor="y", constrain="domain"),
                      yaxis=dict(title="y (mm), height"), height=max(320, int(900 * mesh.H / mesh.W) + 160))
    return fig


def mesh_figure(mesh: Mesh, max_edges=60000):
    p, t = mesh.p * 1e3, mesh.t
    colors = {REG_CASING: "#9aa4b1", REG_FILLER: "#e0b25c"}
    fig = go.Figure()
    for reg_name, mask, col in [("Casing (Al)", mesh.region == REG_CASING, "#7d8794"),
                                ("Filler", mesh.region == REG_FILLER, "#c99a3c"),
                                ("Cells (21700)", mesh.region >= 10, "#2f6db5")]:
        tt = t[mask]
        if len(tt) == 0:
            continue
        e = np.vstack([tt[:, [0, 1]], tt[:, [1, 2]], tt[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        if len(e) > max_edges:
            e = e[np.random.default_rng(0).choice(len(e), max_edges, replace=False)]
        xs = np.c_[p[e[:, 0], 0], p[e[:, 1], 0], np.full(len(e), np.nan)].ravel()
        ys = np.c_[p[e[:, 0], 1], p[e[:, 1], 1], np.full(len(e), np.nan)].ravel()
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=0.5, color=col), name=reg_name,
                                   hoverinfo="skip"))
    fig.update_layout(font=FONT, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="x (mm)", scaleanchor="y", constrain="domain"),
                      yaxis=dict(title="y (mm)"), height=max(320, int(900 * mesh.H / mesh.W) + 160),
                      legend=dict(orientation="h", y=1.08))
    return fig


def cell_bar(cell_mean, cell_max, limit):
    n = len(cell_mean)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(1, n + 1), y=cell_mean, name="Cell mean", marker_color="#2f6db5"))
    fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=cell_max, name="Cell max", mode="markers",
                             marker=dict(color="#d1495b", size=6)))
    fig.add_hline(y=limit, line_dash="dash", line_color="#d1495b", annotation_text=f"limit {limit:.0f} °C")
    lo = float(np.min(cell_mean)) - 2
    fig.update_layout(font=FONT, margin=dict(l=10, r=10, t=30, b=10), height=320,
                      xaxis_title="Cell number (row-major, from bottom-left)", yaxis_title="T (°C)",
                      yaxis_range=[lo, max(limit, float(np.max(cell_max))) + 2],
                      legend=dict(orientation="h", y=1.12))
    return fig


def transient_figure(res, limit):
    t_min = res["t"] / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_min, y=res["Tmax"], name="Peak T (any point)", line=dict(color="#d1495b")))
    fig.add_trace(go.Scatter(x=t_min, y=res["Tcell_hi"], name="Hottest cell (mean)", line=dict(color="#e07a24")))
    fig.add_trace(go.Scatter(x=t_min, y=res["Tcell_lo"], name="Coolest cell (mean)", line=dict(color="#2f6db5")))
    fig.add_trace(go.Scatter(x=t_min, y=res["I"], name="Pack current (A)", yaxis="y2",
                             line=dict(color="rgba(120,120,120,0.6)", width=1)))
    fig.add_hline(y=limit, line_dash="dash", line_color="#d1495b", annotation_text=f"limit {limit:.0f} °C")
    fig.update_layout(font=FONT, height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Time (min)", yaxis_title="Temperature (°C)",
                      yaxis2=dict(title="Pack current (A)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12))
    return fig


def convergence_figure(x, y, xlab, ylab, ref_slope=None, logx=True, logy=True, name="FEM"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color="#2f6db5")))
    if ref_slope is not None:
        x = np.asarray(x, float)
        yref = y[0] * (x / x[0]) ** ref_slope
        fig.add_trace(go.Scatter(x=x, y=yref, mode="lines", name=f"slope {ref_slope}",
                                 line=dict(dash="dash", color="#999")))
    fig.update_layout(font=FONT, height=330, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title=xlab, type="log" if logx else "linear"),
                      yaxis=dict(title=ylab, type="log" if logy else "linear"),
                      legend=dict(orientation="h", y=1.12))
    return fig


def radial_profile_figure(r, T, Tex):
    o = np.argsort(r)
    rr = np.linspace(0, r.max(), 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r[o] * 1e3, y=T[o], mode="markers", name="FEM nodes",
                             marker=dict(size=3, color="#2f6db5")))
    fig.add_trace(go.Scatter(x=r[o] * 1e3, y=Tex[o], mode="lines", name="Analytical", line=dict(color="#d1495b")))
    fig.update_layout(font=FONT, height=330, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Radius r (mm)", yaxis_title="T (°C)", legend=dict(orientation="h", y=1.12))
    return fig
