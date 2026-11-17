# Ather 450X battery pack: 2-D FEA thermal analysis

**RV College of Engineering · Department of Mechanical Engineering · Experiential Learning Phase II · ACY 2026-27**

Raghav Mahawar (1RV23ME086) · Panyam Chetan Sai Anirv (1RV23ME079) · Vishal Kumar Singh (1RV23ME122) · Mahesh Kembhavi (1RV23ME065)
Guide: Dr. Shanmukha Nagaraj (Professor & Head, ME) · Co-Guides: Dr. Anjaneya G, Dr. Kirthan L J

| | |
|---|---|
| Live app | https://ather450x-battery-thermal.streamlit.app |
| Project page | https://fak3eer.github.io/ather450x-battery-thermal-fea/ |
| Video demonstration | https://fak3eer.github.io/ather450x-battery-thermal-fea/video.html |
| Slides (PDF) | https://fak3eer.github.io/ather450x-battery-thermal-fea/slides.pdf |
| Technical paper (PDF) | https://fak3eer.github.io/ather450x-battery-thermal-fea/paper.pdf |

A Streamlit app that solves the temperature field in a cross-section of the Ather 450X battery pack with
the finite element method, at **steady state and over time**. It is verified against analytical and
manufactured solutions, and it exports **ANSYS Mechanical APDL** decks for an independent cross-check.

## What changed from v1 (reviewer feedback → response)

| Feedback | v1 | v2 |
|---|---|---|
| **Governing equations, references** | Laplace + source, Q4 weak form, no citations | Full transient energy equation, cylindrically orthotropic cell conductivity, Bernardi heat generation (irreversible + entropic), Arrhenius R(T), correlation-based forced/natural/mixed convection + radiation, Galerkin T3 matrices, θ-method time stepping. 17 numbered references, cited inline |
| **Actual dimensions** | Assumed 140 × 90 mm homogenised rectangle, "4 prismatic cells" | Real **21700 cells (Ø21.10 × 70.15 mm, 68 g)** meshed as individual circles. **14S12P / 168 cells, 51.1 V, Al casing** from Ather and community sources. Every value is labelled *published* or *assumed* |
| **Analyse through ANSYS** | none | Two APDL input files per case: (A) **same mesh** (PLANE55, code verification) and (B) **ANSYS-meshed geometry** (PLANE77, solution verification). Plus a DXF for Workbench/SpaceClaim, a step-by-step Workbench recipe, and **upload of ANSYS nodal results** for node-by-node comparison |
| **Technical details** | brief | Energy balance check, Picard convergence history, h-breakdown per face, GCI mesh study (Celik 2008), MMS order of accuracy, analytical orthotropic benchmark, transient ride/charge profiles, assumptions & limitations |

## Physics in one screen

```
ρc ∂T/∂t = ∇·(K∇T) + q'''                      (per unit depth)
K_cell = k_r e_r e_rᵀ + k_θ e_θ e_θᵀ            (spiral-wound jelly roll)
Q_cell = I_c² R(T) − I_c T dU/dT,  I_c = I_pack / N_p      (Bernardi 1985)
−n·K∇T = (h_conv + h_rad)(T − T∞)               (Robin), or T = T_D, or insulated
h_conv: flat-plate forced (Incropera 7.30/7.38) ⊕ natural (Churchill–Chu, hot-plate) blended with n = 3.5
I_pack from road load: [mg(Crr cosα + sinα) + ½ρCdA v²] v / η / V_nom
```

## Verification results (default case)

| Check | Result |
|---|---|
| Single orthotropic cell vs analytical (k_r = 0.9, k_θ = 24.2) | L2 error 0.093 → 0.0017 K as h goes 2 → 0.25 mm (≈ 4× per halving, 2nd order) |
| Manufactured solution, observed L2 order | 2.05, 1.92, 1.96 (theory: 2) |
| Global energy balance | residual ~1e-5 % |
| Pack mesh study, GCI (Celik et al. 2008) | p = 1.73, fine-grid GCI ≈ 0.2 % on peak T |
| Exported APDL deck re-solved by **scikit-fem** (independent FE code) | max |ΔT| = 7e-8 K (all BC types) |

Run them with `python -m pytest -q tests` (needs `pytest` and `scikit-fem`).

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Streamlit Community Cloud**: push this folder to a GitHub repo, then pick `app.py` as the entry point.
`requirements.txt` covers everything; `triangle` ships binary wheels, so no compiler is needed.

## ANSYS workflow

1. Set up the case in the app, then open the **ANSYS cross-check** tab.
2. Download `ather450x_same_mesh.inp` (and/or `ather450x_ansys_mesh.inp`).
3. In Mechanical APDL: *File → Read Input from…*. The deck solves, plots the field and writes
   `ansys_nodal_T_*.csv`.
4. Upload the CSV in the same tab to get peak-T difference, max/RMS nodal ΔT and a parity plot.

Expected: deck A agrees to round-off (same discretisation). Deck B agrees within the discretisation error.
Check the cell orthotropy with `/PSYMB,ESYS,1` + `EPLOT`; each cell uses a local cylindrical CS (`LOCAL,100+i,1`).

> The APDL decks were generated and checked by re-solving deck A with an independent FE code. They have not
> been run inside ANSYS by the author of this app, so please report any command that your ANSYS version rejects.

## Project layout

```
app.py                     Streamlit UI (7 tabs)
core/physics.py            cell data, materials, Bernardi heat, convection/radiation, road load
core/mesh.py               geometry + Triangle meshing (area-preserving cell polygons)
core/fem.py                T3 assembly, orthotropic tensors, steady Picard, backward-Euler transient, energy balance
core/verification.py       analytical cell, MMS, GCI
core/ansys_export.py       APDL decks (same mesh / ANSYS mesh), DXF, CSV import
core/plots.py              Plotly figures
core/references.py         numbered references
tests/                     pytest suite incl. independent scikit-fem solve of the exported deck
```

## Key assumptions (see the app's last tab for the full list)

- Ather does not publish its cell supplier or internal layout. LG INR21700-M50T stands in as a
  representative 21700 NMC cell, and the section layout (3 × 14 cells), holder pitch (22.5 mm), wall (3 mm)
  and filler are editable assumptions.
- The 2-D section neglects axial heat flow to the end plates and busbars, so results are conservative.
- Steady state is the long-duration bound. Use the transient tab for a real ride or charge.
