"""
Finite element solver for 2-D heat conduction with linear (T3) triangles.

Strong form (per unit depth):
    rho c dT/dt = div( K grad T ) + q'''          in Omega
    -n.(K grad T) = h (T - T_inf)                 on Gamma_R   (convection + radiation)
    T = T_D                                       on Gamma_D
    -n.(K grad T) = 0                             on Gamma_N   (insulated)

Inside each cylindrical cell the conductivity tensor is cylindrically orthotropic
(spiral-wound jelly roll): K = k_r e_r e_r^T + k_t e_t e_t^T, evaluated at the element
centroid relative to the cell axis.

Galerkin weak form and element matrices follow [12][13]:
    K_e = A_e B^T K B,  B = (1/2A_e) [[b1 b2 b3],[c1 c2 c3]]
    M_e = rho c A_e/12 [[2,1,1],[1,2,1],[1,1,2]]       (consistent capacity)
    f_e = q''' A_e/3 [1,1,1]
    H_e = h L/6 [[2,1],[1,2]],  r_e = h T_inf L/2 [1,1]  (Robin edge)
Time integration: theta-method (theta = 1, backward Euler) [13].
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh, REG_CASING, REG_FILLER, REG_CELL0


# --------------------------------------------------------------------------
# Element-level quantities
# --------------------------------------------------------------------------
def _geom(mesh: Mesh):
    p, t = mesh.p, mesh.t
    x = p[t, 0]
    y = p[t, 1]
    b = np.c_[y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]]
    c = np.c_[x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]]
    A = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    return b, c, A


def conductivity_tensors(mesh: Mesh, k_cell_r, k_cell_t, k_filler, k_casing):
    """Per-element 2x2 conductivity tensors (ne, 2, 2)."""
    ne = mesh.n_elems
    Kt = np.zeros((ne, 2, 2))
    reg = mesh.region
    iso = np.where(reg == REG_CASING, k_casing, np.where(reg == REG_FILLER, k_filler, 0.0))
    Kt[:, 0, 0] = iso
    Kt[:, 1, 1] = iso
    cells = reg >= REG_CELL0
    if cells.any():
        cid = reg[cells] - REG_CELL0
        cen = mesh.centroids()[cells]
        d = cen - mesh.centres[cid]
        nrm = np.linalg.norm(d, axis=1)
        nrm[nrm == 0] = 1.0
        er = d / nrm[:, None]
        et = np.c_[-er[:, 1], er[:, 0]]
        Kt[cells] = (k_cell_r * er[:, :, None] * er[:, None, :]
                     + k_cell_t * et[:, :, None] * et[:, None, :])
    return Kt


def assemble_stiffness(mesh: Mesh, Kt: np.ndarray) -> sp.csr_matrix:
    b, c, A = _geom(mesh)
    B = np.stack([b, c], axis=1) / (2 * A)[:, None, None]     # (ne,2,3)
    Ke = A[:, None, None] * np.einsum("eki,ekl,elj->eij", B, Kt, B)
    t = mesh.t
    rows = np.repeat(t, 3, axis=1).ravel()
    cols = np.tile(t, (1, 3)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))


def assemble_capacity(mesh: Mesh, rhoc_elem: np.ndarray) -> sp.csr_matrix:
    _, _, A = _geom(mesh)
    m = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    Me = (rhoc_elem * A)[:, None, None] * m[None]
    t = mesh.t
    rows = np.repeat(t, 3, axis=1).ravel()
    cols = np.tile(t, (1, 3)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(n, n))


def assemble_source(mesh: Mesh, q_elem: np.ndarray) -> np.ndarray:
    _, _, A = _geom(mesh)
    f = np.zeros(mesh.n_nodes)
    np.add.at(f, mesh.t.ravel(), np.repeat(q_elem * A / 3.0, 3))
    return f


def robin_terms(mesh: Mesh, edges: np.ndarray, h: float, T_inf: float):
    n = mesh.n_nodes
    if len(edges) == 0 or h == 0:
        return sp.csr_matrix((n, n)), np.zeros(n)
    L = np.linalg.norm(mesh.p[edges[:, 0]] - mesh.p[edges[:, 1]], axis=1)
    he = (h * L / 6.0)[:, None, None] * np.array([[2, 1], [1, 2]])[None]
    rows = np.repeat(edges, 2, axis=1).ravel()
    cols = np.tile(edges, (1, 2)).ravel()
    H = sp.csr_matrix((he.ravel(), (rows, cols)), shape=(n, n))
    r = np.zeros(n)
    np.add.at(r, edges.ravel(), np.repeat(h * T_inf * L / 2.0, 2))
    return H, r


def edge_mean(mesh: Mesh, edges: np.ndarray, T: np.ndarray) -> float:
    if len(edges) == 0:
        return float("nan")
    L = np.linalg.norm(mesh.p[edges[:, 0]] - mesh.p[edges[:, 1]], axis=1)
    return float(np.sum(L * 0.5 * (T[edges[:, 0]] + T[edges[:, 1]])) / L.sum())


def solve_dirichlet(Kg: sp.csr_matrix, f: np.ndarray, fixed: np.ndarray, vals: np.ndarray):
    """Direct elimination of Dirichlet DOFs, sparse LU solve."""
    n = Kg.shape[0]
    T = np.zeros(n)
    if len(fixed) == 0:
        return spla.spsolve(Kg.tocsc(), f)
    free = np.setdiff1d(np.arange(n), fixed)
    T[fixed] = vals
    rhs = f[free] - Kg[free][:, fixed] @ vals
    T[free] = spla.spsolve(Kg[free][:, free].tocsc(), rhs)
    return T


# --------------------------------------------------------------------------
# Generic linear solve used by the validation cases
# --------------------------------------------------------------------------
def solve_linear(mesh: Mesh, Kt, q_elem, robin=(), dirichlet=()):
    """robin: iterable of (edges, h, T_inf); dirichlet: iterable of (node_ids, value)."""
    K = assemble_stiffness(mesh, Kt)
    f = assemble_source(mesh, q_elem)
    for edges, h, Tinf in robin:
        H, r = robin_terms(mesh, edges, h, Tinf)
        K = K + H
        f = f + r
    fixed, vals = [], []
    for nodes, v in dirichlet:
        nodes = np.atleast_1d(nodes)
        fixed.append(nodes)
        vals.append(np.broadcast_to(np.asarray(v, dtype=float), nodes.shape))
    if fixed:
        fx = np.concatenate(fixed)
        vv = np.concatenate(vals)
        fx, idx = np.unique(fx, return_index=True)
        vv = vv[idx]
    else:
        fx, vv = np.array([], int), np.array([])
    return solve_dirichlet(K.tocsr(), f, fx, vv)


# --------------------------------------------------------------------------
# Pack model
# --------------------------------------------------------------------------
@dataclass
class PackProblem:
    mesh: Mesh
    k_cell_r: float
    k_cell_t: float
    k_filler: float
    k_casing: float
    rhoc_cell: float
    rhoc_filler: float
    rhoc_casing: float
    cell_area_true: float           # pi r^2, m2
    cell_length: float              # m (for q''' = Q / V_cell)
    side_bc: dict                   # side -> physics.SideBC
    h_func: callable                # (side, bc, T_surf_C) -> dict with 'h'
    q_cell_func: callable           # (T_cell_mean_C array, I_cell) -> W per cell
    T_inf: float

    def __post_init__(self):
        m = self.mesh
        self.Kt = conductivity_tensors(m, self.k_cell_r, self.k_cell_t, self.k_filler, self.k_casing)
        self.K0 = assemble_stiffness(m, self.Kt)
        reg = m.region
        rhoc = np.where(reg == REG_CASING, self.rhoc_casing,
                        np.where(reg == REG_FILLER, self.rhoc_filler, self.rhoc_cell))
        self.M = assemble_capacity(m, rhoc)
        _, _, self.Ae = _geom(m)
        self.cell_of_elem = np.where(reg >= REG_CELL0, reg - REG_CELL0, -1)
        self.n_cells = len(m.centres)
        self.cell_area_mesh = np.bincount(self.cell_of_elem[self.cell_of_elem >= 0],
                                          weights=self.Ae[self.cell_of_elem >= 0],
                                          minlength=self.n_cells)
        fixed, vals = [], []
        for side, bc in self.side_bc.items():
            if bc.kind == "fixed":
                nodes = np.unique(m.bedges[side])
                fixed.append(nodes)
                vals.append(np.full(len(nodes), bc.T_fixed_C))
        if fixed:
            fx = np.concatenate(fixed)
            vv = np.concatenate(vals)
            self.fixed, idx = np.unique(fx, return_index=True)
            self.fixed_vals = vv[idx]
        else:
            self.fixed = np.array([], dtype=int)
            self.fixed_vals = np.array([])

    # -- helpers ------------------------------------------------------------
    def cell_means(self, T):
        Te = T[self.mesh.t].mean(axis=1)
        sel = self.cell_of_elem >= 0
        s = np.bincount(self.cell_of_elem[sel], weights=(Te * self.Ae)[sel], minlength=self.n_cells)
        return s / self.cell_area_mesh

    def cell_max(self, T):
        out = np.full(self.n_cells, -np.inf)
        sel = self.cell_of_elem >= 0
        Tn = T[self.mesh.t].max(axis=1)
        np.maximum.at(out, self.cell_of_elem[sel], Tn[sel])
        return out

    def q_elem_from_cells(self, Q_cell):
        """Volumetric generation q''' [W/m3] per element from per-cell heat [W]."""
        q_cell = Q_cell / (self.cell_area_true * self.cell_length)
        q = np.zeros(self.mesh.n_elems)
        sel = self.cell_of_elem >= 0
        q[sel] = q_cell[self.cell_of_elem[sel]]
        return q

    def side_hs(self, T):
        hs = {}
        for side, bc in self.side_bc.items():
            if bc.kind in ("airflow", "natural"):
                Ts = edge_mean(self.mesh, self.mesh.bedges[side], T)
                hs[side] = self.h_func(side, bc, Ts)
            else:
                hs[side] = dict(h=0.0, h_conv=0.0, h_rad=0.0, h_forced=0.0, h_natural=0.0, Re=0.0, Ra=0.0)
        return hs

    def system(self, hs):
        K = self.K0.copy()
        r = np.zeros(self.mesh.n_nodes)
        for side, d in hs.items():
            if d["h"] > 0:
                H, rr = robin_terms(self.mesh, self.mesh.bedges[side], d["h"], self.T_inf)
                K = K + H
                r = r + rr
        return K.tocsr(), r

    # -- steady state with Picard iteration on h(T) and R(T) -----------------
    def solve_steady(self, I_cell, tol=1e-4, maxit=40, T0=None):
        T = np.full(self.mesh.n_nodes, self.T_inf + 10.0) if T0 is None else T0.copy()
        hist = []
        for it in range(maxit):
            hs = self.side_hs(T)
            Q = self.q_cell_func(self.cell_means(T), I_cell)
            K, r = self.system(hs)
            f = assemble_source(self.mesh, self.q_elem_from_cells(Q)) + r
            Tn = solve_dirichlet(K, f, self.fixed, self.fixed_vals)
            dT = float(np.max(np.abs(Tn - T)))
            hist.append(dict(iteration=it + 1, max_change_K=dT, T_max=float(Tn.max())))
            # under-relaxation for robustness of the natural-convection loop
            T = Tn if it < 1 else 0.3 * T + 0.7 * Tn
            if dT < tol:
                T = Tn
                break
        hs = self.side_hs(T)
        Q = self.q_cell_func(self.cell_means(T), I_cell)
        return T, hs, Q, hist

    # -- transient (backward Euler; h and R lagged one step, refactorised when h drifts)
    def solve_transient(self, I_of_t, t_end, dt, T_init, theta=1.0, store_every=1):
        n = self.mesh.n_nodes
        T = np.full(n, T_init, dtype=float)
        if len(self.fixed):
            T[self.fixed] = self.fixed_vals
        free = np.setdiff1d(np.arange(n), self.fixed)
        hs = self.side_hs(T)
        h_ref = {s: d["h"] for s, d in hs.items()}

        def factor(hs):
            K, r = self.system(hs)
            A = (self.M / dt + theta * K).tocsr()
            lu = spla.splu(A[free][:, free].tocsc())
            return K, r, A, lu

        K, r, A, lu = factor(hs)
        times, Tmax, Tcell_hi, Tcell_lo, Qtot, Ilog = [0.0], [T.max()], [], [], [], []
        cm = self.cell_means(T)
        Tcell_hi.append(cm.max())
        Tcell_lo.append(cm.min())
        Q = self.q_cell_func(cm, I_of_t(0.0))
        Qtot.append(Q.sum())
        Ilog.append(I_of_t(0.0))
        snaps = [(0.0, T.copy())]
        nsteps = int(np.ceil(t_end / dt))
        for k in range(1, nsteps + 1):
            tn = k * dt
            I = I_of_t(tn - 0.5 * dt)
            Q = self.q_cell_func(self.cell_means(T), I)
            f = assemble_source(self.mesh, self.q_elem_from_cells(Q)) + r
            rhs = (self.M / dt) @ T - (1 - theta) * (K @ T) + f
            if len(self.fixed):
                rhs_free = rhs[free] - A[free][:, self.fixed] @ self.fixed_vals
            else:
                rhs_free = rhs
            Tn = T.copy()
            Tn[free] = lu.solve(rhs_free)
            T = Tn
            # update h if any side has drifted by > 3 %
            hs_new = self.side_hs(T)
            drift = max((abs(hs_new[s]["h"] - h_ref[s]) / max(h_ref[s], 1e-9)) for s in hs_new
                        if self.side_bc[s].kind in ("airflow", "natural")) if any(
                self.side_bc[s].kind in ("airflow", "natural") for s in hs_new) else 0.0
            if drift > 0.03:
                hs = hs_new
                h_ref = {s: d["h"] for s, d in hs.items()}
                K, r, A, lu = factor(hs)
            cm = self.cell_means(T)
            times.append(tn)
            Tmax.append(T.max())
            Tcell_hi.append(cm.max())
            Tcell_lo.append(cm.min())
            Qtot.append(Q.sum())
            Ilog.append(I)
            if k % store_every == 0 or k == nsteps:
                snaps.append((tn, T.copy()))
        return dict(t=np.array(times), Tmax=np.array(Tmax), Tcell_hi=np.array(Tcell_hi),
                    Tcell_lo=np.array(Tcell_lo), Q=np.array(Qtot), I=np.array(Ilog),
                    snaps=snaps, T=T, hs=hs)

    # -- energy balance check ------------------------------------------------
    def energy_balance(self, T, hs, Q_cell):
        """Heat generated vs heat leaving through all boundaries, per metre depth [W/m]."""
        q_gen = float(np.sum(Q_cell / (self.cell_area_true * self.cell_length) * self.cell_area_mesh))
        out = {}
        for side, d in hs.items():
            e = self.mesh.bedges[side]
            if d["h"] > 0 and len(e):
                L = np.linalg.norm(self.mesh.p[e[:, 0]] - self.mesh.p[e[:, 1]], axis=1)
                Tm = 0.5 * (T[e[:, 0]] + T[e[:, 1]])
                out[side] = float(np.sum(d["h"] * L * (Tm - self.T_inf)))
            else:
                out[side] = 0.0
        # Dirichlet reaction: R = K T - f at the fixed nodes is the heat ENTERING there,
        # so the heat leaving through a fixed-temperature face is -R (consistent nodal flux).
        if len(self.fixed):
            K, r = self.system(hs)
            f = assemble_source(self.mesh, self.q_elem_from_cells(Q_cell)) + r
            react = (K @ T - f)[self.fixed]
            taken = np.zeros(len(self.fixed), dtype=bool)
            for side, bc in self.side_bc.items():
                if bc.kind == "fixed":
                    nodes = np.unique(self.mesh.bedges[side])
                    mask = np.isin(self.fixed, nodes) & ~taken
                    taken |= mask
                    out[side] = float(-react[mask].sum())
        q_out = sum(out.values())
        return dict(q_gen=q_gen, q_out=q_out, by_side=out,
                    residual_pct=100.0 * (q_gen - q_out) / q_gen if q_gen else 0.0)


# --------------------------------------------------------------------------
# Interpolation of the FE field at arbitrary points (for ANSYS comparison)
# --------------------------------------------------------------------------
def interpolate(mesh: Mesh, T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    tri = Triangulation(mesh.p[:, 0], mesh.p[:, 1], mesh.t)
    it = LinearTriInterpolator(tri, T)
    v = it(pts[:, 0], pts[:, 1])
    return np.ma.filled(v, np.nan)
