"""
Geometry and unstructured triangular meshing of the battery cross-section.

The 2-D section is cut normal to the cell axes, so every 21700 cell appears as
a circle of its true diameter. Regions:
    casing (aluminium wall)  -> region id 0
    filler (holder / potting / air between cells) -> region id 1
    cell i                   -> region id 10 + i

Meshing uses Shewchuk's Triangle [15] (constrained conforming Delaunay with
minimum-angle quality constraint).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import triangle as tr

REG_CASING = 0
REG_FILLER = 1
REG_CELL0 = 10


@dataclass
class PackGeometry:
    cell_d: float = 21.10e-3       # m, cell diameter (datasheet)
    pitch: float = 22.5e-3         # m, centre-to-centre spacing in the holder
    rows: int = 3
    cols: int = 14
    layout: str = "aligned"        # 'aligned' or 'staggered'
    clearance: float = 2.0e-3      # m, filler gap between outermost cells and casing wall
    wall: float = 3.0e-3           # m, casing wall thickness

    def validate(self):
        if self.pitch <= self.cell_d + 0.2e-3:
            raise ValueError("Pitch must exceed the cell diameter by at least 0.2 mm.")
        if self.clearance < 0.5e-3:
            raise ValueError("Clearance to the casing must be at least 0.5 mm.")
        if self.wall < 0.5e-3:
            raise ValueError("Casing wall must be at least 0.5 mm.")

    @property
    def row_pitch(self) -> float:
        return self.pitch * (math.sqrt(3) / 2 if self.layout == "staggered" else 1.0)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def inner_w(self) -> float:
        extra = 0.5 * self.pitch if (self.layout == "staggered" and self.rows > 1) else 0.0
        return 2 * self.clearance + self.cell_d + (self.cols - 1) * self.pitch + extra

    @property
    def inner_h(self) -> float:
        return 2 * self.clearance + self.cell_d + (self.rows - 1) * self.row_pitch

    @property
    def W(self) -> float:
        return self.inner_w + 2 * self.wall

    @property
    def H(self) -> float:
        return self.inner_h + 2 * self.wall

    def centres(self) -> np.ndarray:
        r = 0.5 * self.cell_d
        x0 = self.wall + self.clearance + r
        y0 = self.wall + self.clearance + r
        pts = []
        for j in range(self.rows):
            off = 0.5 * self.pitch if (self.layout == "staggered" and j % 2 == 1) else 0.0
            for i in range(self.cols):
                pts.append((x0 + off + i * self.pitch, y0 + j * self.row_pitch))
        return np.array(pts)


@dataclass
class Mesh:
    p: np.ndarray            # (nn, 2) node coordinates [m]
    t: np.ndarray            # (ne, 3) triangle connectivity
    region: np.ndarray       # (ne,) region id per element
    bedges: dict             # side name -> (nb, 2) boundary edge node pairs
    centres: np.ndarray      # (ncell, 2)
    radius: float
    W: float
    H: float

    @property
    def n_nodes(self):
        return self.p.shape[0]

    @property
    def n_elems(self):
        return self.t.shape[0]

    def elem_area(self):
        p, t = self.p, self.t
        x1, y1 = p[t[:, 0]].T
        x2, y2 = p[t[:, 1]].T
        x3, y3 = p[t[:, 2]].T
        return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def centroids(self):
        return self.p[self.t].mean(axis=1)


def _circle_poly(cx, cy, r, n):
    """Polygon with n vertices whose AREA equals that of the true circle
    (vertex radius scaled so heat input and heat capacity are not under-counted)."""
    r_eff = r * math.sqrt(2 * math.pi / (n * math.sin(2 * math.pi / n)))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + r_eff * np.cos(a), cy + r_eff * np.sin(a)]


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _add_loop(V, S, pts):
    n0 = len(V)
    V.extend(pts.tolist())
    k = len(pts)
    S.extend([[n0 + i, n0 + (i + 1) % k] for i in range(k)])


def _orient(p, t):
    x1, y1 = p[t[:, 0]].T
    x2, y2 = p[t[:, 1]].T
    x3, y3 = p[t[:, 2]].T
    a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    t = t.copy()
    neg = a < 0
    t[neg, 1], t[neg, 2] = t[neg, 2].copy(), t[neg, 1].copy()
    return t


def boundary_edges(t: np.ndarray) -> np.ndarray:
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, idx, cnt = np.unique(es, axis=0, return_index=True, return_counts=True)
    return e[idx[cnt == 1]]


def classify_rect_edges(p, be, W, H, tol=1e-9):
    out = {}
    xa, ya = p[be[:, 0]].T
    xb, yb = p[be[:, 1]].T
    out["bottom"] = be[(abs(ya) < tol) & (abs(yb) < tol)]
    out["top"] = be[(abs(ya - H) < tol) & (abs(yb - H) < tol)]
    out["left"] = be[(abs(xa) < tol) & (abs(xb) < tol)]
    out["right"] = be[(abs(xa - W) < tol) & (abs(xb - W) < tol)]
    return out


def mesh_pack(g: PackGeometry, h: float, n_circ: int | None = None) -> Mesh:
    """Mesh the pack section with a target element size h [m]."""
    g.validate()
    r = 0.5 * g.cell_d
    if n_circ is None:
        n_circ = int(max(24, min(96, math.ceil(2 * math.pi * r / h))))
    C = g.centres()
    V, S = [], []
    _add_loop(V, S, _rect(0, 0, g.W, g.H))
    _add_loop(V, S, _rect(g.wall, g.wall, g.W - g.wall, g.H - g.wall))
    for cx, cy in C:
        _add_loop(V, S, _circle_poly(cx, cy, r, n_circ))
    regions = [[0.5 * g.wall, 0.5 * g.wall, REG_CASING, 0],
               [g.wall + 0.5 * g.clearance, g.wall + 0.5 * g.clearance, REG_FILLER, 0]]
    for i, (cx, cy) in enumerate(C):
        regions.append([cx, cy, REG_CELL0 + i, 0])
    A = dict(vertices=np.array(V), segments=np.array(S), regions=np.array(regions, dtype=float))
    amax = h * h * math.sqrt(3) / 4
    out = tr.triangulate(A, f"pq30Aa{amax:.15f}")
    p = out["vertices"]
    t = _orient(p, out["triangles"].astype(np.int64))
    reg = out["triangle_attributes"][:, 0].round().astype(int)
    be = boundary_edges(t)
    bed = classify_rect_edges(p, be, g.W, g.H, tol=1e-9 * max(g.W, g.H) + 1e-12)
    return Mesh(p=p, t=t, region=reg, bedges=bed, centres=C, radius=r, W=g.W, H=g.H)


def mesh_single_cell(r: float, h: float, n_circ: int | None = None) -> Mesh:
    """Single circular cell centred at the origin (for the analytical validation case)."""
    if n_circ is None:
        n_circ = int(max(24, min(256, math.ceil(2 * math.pi * r / h))))
    poly = _circle_poly(0.0, 0.0, r, n_circ)
    V, S = [], []
    _add_loop(V, S, poly)
    A = dict(vertices=np.array(V), segments=np.array(S))
    amax = h * h * math.sqrt(3) / 4
    out = tr.triangulate(A, f"pq30a{amax:.15f}")
    p = out["vertices"]
    t = _orient(p, out["triangles"].astype(np.int64))
    reg = np.full(t.shape[0], REG_CELL0)
    be = boundary_edges(t)
    return Mesh(p=p, t=t, region=reg, bedges={"perimeter": be},
                centres=np.array([[0.0, 0.0]]), radius=r, W=2 * r, H=2 * r)


def mesh_unit_square(h: float) -> Mesh:
    V, S = [], []
    _add_loop(V, S, _rect(0, 0, 1, 1))
    A = dict(vertices=np.array(V), segments=np.array(S))
    amax = h * h * math.sqrt(3) / 4
    out = tr.triangulate(A, f"pq30a{amax:.15f}")
    p = out["vertices"]
    t = _orient(p, out["triangles"].astype(np.int64))
    be = boundary_edges(t)
    return Mesh(p=p, t=t, region=np.full(t.shape[0], REG_FILLER),
                bedges=classify_rect_edges(p, be, 1.0, 1.0, 1e-12),
                centres=np.zeros((0, 2)), radius=0.0, W=1.0, H=1.0)
