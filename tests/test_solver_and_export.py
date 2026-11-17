"""
Regression / verification tests.

    python -m pytest -q tests

test_apdl_deck_independent_solve:
    Parses the generated *same-mesh* APDL deck (N, E, MAT, ESYS, LOCAL, ESEL/BFE, NSEL/SF, D)
    and solves it with scikit-fem, a completely independent FE library. Agreement with the
    in-app solver verifies both the solver and the ANSYS export.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.physics import LG_M50T, FILLERS, CASINGS, SideBC, Environment, side_h, cell_heat
from core.mesh import PackGeometry, mesh_pack
from core.fem import PackProblem
from core import verification as ver
from core.ansys_export import same_mesh_apdl, geometry_apdl, geometry_dxf, parse_ansys_csv


def _problem(h=2e-3, sides=None, I_cell=5.0):
    g = PackGeometry(rows=3, cols=6)
    env = Environment(T_amb_C=35.0, speed_kmh=50.0)
    cell = LG_M50T
    fil = FILLERS["Cell holder plastic (PC/ABS)"]
    cas = CASINGS["Aluminium alloy, die-cast (ADC12-type)"]
    sides = sides or {"bottom": SideBC("airflow"), "top": SideBC("fixed", 30.0),
                      "left": SideBC("natural"), "right": SideBC("insulated")}
    prob = PackProblem(mesh_pack(g, h), cell.k_radial, cell.k_axial, fil.k, cas.k,
                       cell.density * cell.cp, fil.rho * fil.cp, cas.rho * cas.cp,
                       math.pi * cell.radius ** 2, cell.height, sides,
                       lambda s, bc, Ts: side_h(s, bc, Ts, env, g.W, g.H),
                       lambda Tc, Ic: cell_heat(Ic, Tc, cell), env.T_amb_C)
    T, hs, Q, hist = prob.solve_steady(I_cell)
    mat = dict(k_r=cell.k_radial, k_t=cell.k_axial, rho_cell=cell.density, cp_cell=cell.cp,
               k_fill=fil.k, rho_fill=fil.rho, cp_fill=fil.cp, k_case=cas.k, rho_case=cas.rho, cp_case=cas.cp)
    return g, env, prob, T, hs, Q, mat, sides


def test_analytical_single_cell():
    rows, _ = ver.single_cell_case(hs=(1e-3, 5e-4, 2.5e-4))
    errs = [r["L2_err_K"] for r in rows]
    assert errs[-1] < 0.005
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0   # ~second order


def test_mms_order():
    rows = ver.mms_case()
    assert 1.8 < rows[-1]["order"] < 2.2


def test_energy_balance():
    _, _, prob, T, hs, Q, _, _ = _problem()
    eb = prob.energy_balance(T, hs, Q)
    assert abs(eb["residual_pct"]) < 1e-6


def _parse_deck(text):
    nodes, elems, mats, esys, local, hgen, conv, fixed = {}, [], [], [], {}, {}, [], []
    cur_mat, cur_esys = 1, 0
    esel = None
    nsel = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("!"):
            continue
        f = ln.split(",")
        c = f[0].upper()
        if c == "N":
            nodes[int(f[1])] = (float(f[2]), float(f[3]))
        elif c == "E":
            elems.append([int(f[1]), int(f[2]), int(f[3])])
            mats.append(cur_mat)
            esys.append(cur_esys)
        elif c == "MAT":
            cur_mat = int(f[1])
        elif c == "ESYS":
            cur_esys = int(f[1])
        elif c == "LOCAL":
            local[int(f[1])] = (float(f[3]), float(f[4]))
        elif c == "ESEL":
            esel = (int(f[4]), int(f[5]))
        elif c == "BFE":
            for e in range(esel[0], esel[1] + 1):
                hgen[e] = float(f[4])
        elif c == "NSEL" and len(f) > 5 and f[2].upper() == "LOC":
            nsel = (f[3].upper(), float(f[4]), float(f[5]))
        elif c == "SF":
            conv.append((nsel, float(f[3]), float(f[4])))
        elif c == "D" and len(f) > 3 and f[2].upper() == "TEMP":
            fixed.append((nsel, float(f[3])))
    return nodes, np.array(elems) - 1, np.array(mats), np.array(esys), local, hgen, conv, fixed


def test_apdl_deck_independent_solve():
    import skfem as fem
    from skfem.helpers import dot, grad

    g, env, prob, T_app, hs, Q, mat, sides = _problem(h=2e-3)
    q_cell = Q / (math.pi * LG_M50T.radius ** 2 * LG_M50T.height)
    deck = same_mesh_apdl(prob.mesh, mat, hs, {s: b.kind for s, b in sides.items()},
                          {s: b.T_fixed_C for s, b in sides.items()}, q_cell, env.T_amb_C)
    nodes, t, mats, esys, local, hgen, conv, fixed = _parse_deck(deck)
    p = np.array([nodes[i + 1] for i in range(len(nodes))])
    m = fem.MeshTri(p.T, t.T)
    # per-element tensor from the deck: MAT 1 orthotropic in its LOCAL cylindrical system
    cen = p[t].mean(axis=1)
    kxx = np.zeros(len(t)); kyy = np.zeros(len(t)); kxy = np.zeros(len(t))
    kiso = {2: mat["k_fill"], 3: mat["k_case"]}
    for e in range(len(t)):
        if mats[e] == 1:
            xc, yc = local[esys[e]]
            d = cen[e] - (xc, yc)
            er = d / np.linalg.norm(d)
            et = np.array([-er[1], er[0]])
            K = mat["k_r"] * np.outer(er, er) + mat["k_t"] * np.outer(et, et)
            kxx[e], kxy[e], kyy[e] = K[0, 0], K[0, 1], K[1, 1]
        else:
            kxx[e] = kyy[e] = kiso[mats[e]]
    q = np.array([hgen.get(e + 1, 0.0) for e in range(len(t))])
    basis = fem.Basis(m, fem.ElementTriP1())
    b0 = basis.with_element(fem.ElementTriP0())
    KXX, KXY, KYY, QQ = (b0.interpolate(v) for v in (kxx, kxy, kyy, q))

    @fem.BilinearForm
    def a(u, v, w):
        gu, gv = grad(u), grad(v)
        return (w.kxx * gu[0] + w.kxy * gu[1]) * gv[0] + (w.kxy * gu[0] + w.kyy * gu[1]) * gv[1]

    @fem.LinearForm
    def lf(v, w):
        return w.q * v

    A = a.assemble(basis, kxx=KXX, kxy=KXY, kyy=KYY)
    b = lf.assemble(basis, q=QQ)

    def sel(spec):
        ax, lo, hi = spec
        col = 0 if ax == "X" else 1
        return np.where((p[:, col] >= lo) & (p[:, col] <= hi))[0]

    for spec, h, Tinf in conv:
        nset = set(sel(spec).tolist())
        facets = [f for f in m.boundary_facets() if set(m.facets[:, f].tolist()) <= nset]
        fb = fem.FacetBasis(m, fem.ElementTriP1(), facets=np.array(facets))

        @fem.BilinearForm
        def ah(u, v, w):
            return h * u * v

        @fem.LinearForm
        def lh(v, w):
            return h * Tinf * v

        A = A + ah.assemble(fb)
        b = b + lh.assemble(fb)
    D = np.concatenate([sel(spec) for spec, _ in fixed]) if fixed else np.array([], int)
    x = np.zeros(len(p))
    for spec, val in fixed:
        x[sel(spec)] = val
    T_sk = fem.solve(*fem.condense(A, b, x=x, D=D))
    err = np.max(np.abs(T_sk - T_app))
    print("max |T_skfem - T_app| =", err, "Tmax", T_app.max(), T_sk.max())
    assert err < 1e-6


def test_geometry_deck_and_dxf_render():
    g, env, prob, T, hs, Q, mat, sides = _problem(h=3e-3)
    q_cell = Q / (math.pi * LG_M50T.radius ** 2 * LG_M50T.height)
    deck = geometry_apdl(g, mat, hs, {s: b.kind for s, b in sides.items()},
                         {s: b.T_fixed_C for s, b in sides.items()}, q_cell, env.T_amb_C, esize=2e-3)
    assert deck.count("CYL4,") == g.n_cells
    assert deck.count("AATT,1,,1,") == g.n_cells
    assert "ASBA," in deck and "AGLUE,ALL" in deck and "AMESH,ALL" in deck
    dxf = geometry_dxf(g)
    assert dxf.count("CIRCLE") == g.n_cells and dxf.strip().endswith("EOF")


def test_parse_ansys_csv():
    txt = "        1.,  0.1000000000E-01,  0.2000000000E-01,  0.4012345678E+02,  1.\n" \
          "        2.,  0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00, -1.\n"
    arr = parse_ansys_csv(txt)
    assert arr.shape == (1, 3) and abs(arr[0, 2] - 40.12345678) < 1e-9


if __name__ == "__main__":
    test_apdl_deck_independent_solve()
