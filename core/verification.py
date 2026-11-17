"""
Code and solution verification.

1. Analytical benchmark: single cylindrical cell, uniform q''', convection h at r = R,
   cylindrically orthotropic conductivity (k_r, k_t). Exact solution (axisymmetric, so
   only k_r enters):
        T(r) = T_inf + q R / (2 h) + q (R^2 - r^2) / (4 k_r)
   This checks the orthotropic tensor rotation, the source term and the Robin terms.

2. Method of manufactured solutions (MMS) [16] on the unit square with Dirichlet data:
        T_ex = sin(pi x) sin(pi y),  q''' = 2 pi^2 k sin(pi x) sin(pi y)
   Observed L2 order of accuracy should approach 2 for linear triangles.

3. Grid Convergence Index (Celik et al. 2008 [14]) for the pack peak temperature.
"""
from __future__ import annotations

import math
import numpy as np

from .mesh import mesh_single_cell, mesh_unit_square
from .fem import conductivity_tensors, solve_linear, _geom


def single_cell_case(R=10.55e-3, k_r=0.9, k_t=24.2, q=2.0e5, h=15.0, T_inf=30.0, hs=(2e-3, 1e-3, 5e-4, 2.5e-4)):
    rows = []
    best = None
    for hh in hs:
        m = mesh_single_cell(R, hh)
        Kt = conductivity_tensors(m, k_r, k_t, 1.0, 1.0)
        qe = np.full(m.n_elems, q)
        T = solve_linear(m, Kt, qe, robin=[(m.bedges["perimeter"], h, T_inf)])
        r = np.linalg.norm(m.p, axis=1)
        Tex = T_inf + q * R / (2 * h) + q * (R ** 2 - r ** 2) / (4 * k_r)
        # L2 error (nodal quadrature, area-weighted) and max nodal error
        _, _, A = _geom(m)
        e2 = ((T - Tex)[m.t] ** 2).mean(axis=1)
        L2 = math.sqrt(np.sum(e2 * A) / np.sum(A))
        rows.append(dict(h_mm=hh * 1e3, nodes=m.n_nodes, elements=m.n_elems,
                         T_centre_fem=float(T[np.argmin(r)]),
                         T_centre_exact=float(T_inf + q * R / (2 * h) + q * R ** 2 / (4 * k_r)),
                         max_abs_err_K=float(np.max(np.abs(T - Tex))), L2_err_K=L2))
        best = (m, T, Tex, r)
    return rows, best


def mms_case(k=1.0, hs=(0.2, 0.1, 0.05, 0.025)):
    rows = []
    for hh in hs:
        m = mesh_unit_square(hh)
        Kt = conductivity_tensors(m, k, k, k, k)
        c = m.centroids()
        # source integrated with 1-point centroid rule is O(h^2) consistent; use 3-point edge-midpoint rule
        p, t = m.p, m.t
        mids = [(p[t[:, 0]] + p[t[:, 1]]) / 2, (p[t[:, 1]] + p[t[:, 2]]) / 2, (p[t[:, 2]] + p[t[:, 0]]) / 2]
        qmid = [2 * math.pi ** 2 * k * np.sin(math.pi * x[:, 0]) * np.sin(math.pi * x[:, 1]) for x in mids]
        q_elem = sum(qmid) / 3.0
        bn = np.unique(np.vstack(list(m.bedges.values())))
        T = solve_linear(m, Kt, q_elem, dirichlet=[(bn, 0.0)])
        Tex = np.sin(math.pi * p[:, 0]) * np.sin(math.pi * p[:, 1])
        _, _, A = _geom(m)
        e2 = ((T - Tex)[t] ** 2).mean(axis=1)
        L2 = math.sqrt(np.sum(e2 * A))
        h_eff = math.sqrt(A.sum() / m.n_elems)
        rows.append(dict(h_target=hh, h_eff=h_eff, nodes=m.n_nodes, L2_err=L2))
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        b["order"] = math.log(a["L2_err"] / b["L2_err"]) / math.log(a["h_eff"] / b["h_eff"])
    rows[0]["order"] = float("nan")
    return rows


def gci(N, phi, Fs=1.25):
    """Celik et al. (2008) three-grid GCI. N: element counts (coarse->fine), phi: values.
    Returns dict with observed order p, extrapolated value and fine-grid GCI (%)."""
    N = np.asarray(N, float)
    phi = np.asarray(phi, float)
    order = np.argsort(-N)            # fine first
    N1, N2, N3 = N[order]
    f1, f2, f3 = phi[order]
    h1, h2, h3 = N1 ** -0.5, N2 ** -0.5, N3 ** -0.5
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if abs(e21) < 1e-12 or abs(e32) < 1e-12:
        return dict(p=float("nan"), phi_ext=f1, gci_fine_pct=0.0, r21=r21, r32=r32, note="converged to round-off")
    s = np.sign(e32 / e21)
    p = 1.0
    for _ in range(100):
        qp = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + qp) / math.log(r21)
        if abs(p_new - p) < 1e-8:
            break
        p = p_new
    phi_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1)
    gci_fine = Fs * ea21 / (r21 ** p - 1)
    note = "monotonic" if s > 0 else "oscillatory"
    return dict(p=p, phi_ext=phi_ext, gci_fine_pct=100 * gci_fine, r21=r21, r32=r32, note=note)
