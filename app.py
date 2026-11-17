"""
Ather 450X battery pack: 2-D finite element thermal analysis
(steady state + transient), with verification and an ANSYS cross-check workflow.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import streamlit as st

from core.physics import (LG_M50T, CellSpec, FILLERS, CASINGS, SideBC, Environment, Vehicle,
                          side_h, cell_heat, cruise_power, pack_current, cell_resistance)
from core.mesh import PackGeometry, mesh_pack
from core.fem import PackProblem, interpolate
from core import verification as ver
from core import plots
from core.ansys_export import same_mesh_apdl, geometry_apdl, geometry_dxf, parse_ansys_csv
from core.references import ref_markdown

st.set_page_config(page_title="Ather 450X Battery Thermal FEA", page_icon="🔋", layout="wide",
                   initial_sidebar_state="collapsed")

# =============================================================================
# Sidebar inputs
# =============================================================================
sb = st.sidebar
sb.title("Advanced settings")
sb.caption("Pick a scenario and ambient temperature on the main page. Change these only to explore the model.")

with sb.expander("1 · Cell", expanded=False):
    use_custom = st.toggle("Edit cell parameters", value=False,
                           help="Default: LG INR21700-M50T datasheet + measured 21700 thermal properties.")
    c0 = LG_M50T
    if use_custom:
        d_mm = st.number_input("Diameter (mm)", 18.0, 46.0, c0.diameter * 1e3, 0.05)
        h_mm = st.number_input("Height (mm)", 50.0, 120.0, c0.height * 1e3, 0.05)
        m_g = st.number_input("Mass (g)", 30.0, 400.0, c0.mass * 1e3, 0.5)
        cap = st.number_input("Capacity (Ah)", 1.0, 30.0, c0.capacity_Ah, 0.05)
        dcir = st.number_input("DC internal resistance at 25 °C (mΩ)", 2.0, 80.0, c0.dcir * 1e3, 0.5)
        kr = st.number_input("Radial conductivity k_r (W/m·K)", 0.1, 5.0, c0.k_radial, 0.05)
        kt = st.number_input("Along-layer conductivity k_θ ≈ k_axial (W/m·K)", 1.0, 60.0, c0.k_axial, 0.5)
        cp = st.number_input("Specific heat c_p (J/kg·K)", 500.0, 1500.0, c0.cp, 5.0)
        tlim_d = st.number_input("Max discharge temperature (°C)", 40.0, 80.0, c0.t_max_discharge_C, 1.0)
        tlim_c = st.number_input("Max charge temperature (°C)", 30.0, 60.0, c0.t_max_charge_C, 1.0)
        cell = CellSpec("Custom 21700", d_mm * 1e-3, h_mm * 1e-3, m_g * 1e-3, cap, c0.v_nom, dcir * 1e-3, 25.0,
                        kr, kt, cp, 0.0, tlim_d, tlim_c, "user input")
    else:
        cell = c0
        st.caption(f"**{cell.name}** · Ø{cell.diameter*1e3:.2f} × {cell.height*1e3:.2f} mm · "
                   f"{cell.mass*1e3:.0f} g · {cell.capacity_Ah} Ah · DCIR {cell.dcir*1e3:.0f} mΩ · "
                   f"k_r {cell.k_radial} / k_axial {cell.k_axial} W/m·K · c_p {cell.cp:.0f} J/kg·K")
    dUdT_mV = st.number_input("Entropic coefficient dU/dT (mV/K)", -0.6, 0.6, 0.0, 0.05,
                              help="Reversible heat term of Bernardi's balance. NMC/graphite varies roughly "
                                   "-0.4 to +0.1 mV/K with SOC [2]. 0 = ohmic heat only.")
    Ea_kJ = st.number_input("DCIR activation energy Ea (kJ/mol)", 0.0, 60.0, 0.0, 1.0,
                            help="Arrhenius temperature dependence of resistance. 0 = constant R. "
                                 "Fit to your own HPPC data if available (assumption, not a datasheet value).")

with sb.expander("2 · Pack layout and section", expanded=False):
    n_s = st.number_input("Cells in series (S)", 1, 30, 14)
    n_p = st.number_input("Cells in parallel (P)", 1, 30, 12)
    st.caption("14S12P = 168 cells: community-reported for the 450X 2.9 kWh pack [4]; "
               f"gives {n_s*n_p*cell.energy_Wh/1000:.2f} kWh at {n_s*cell.v_nom:.1f} V "
               "(Ather: 2.9 kWh, 51.1 V [3]).")
    rows = st.number_input("Cell rows in section (height)", 1, 8, 3)
    cols = st.number_input("Cell columns in section (length)", 1, 60, 14)
    layout = st.radio("Holder pattern", ["aligned", "staggered"], horizontal=True)
    pitch_mm = st.number_input("Holder pitch (mm)", cell.diameter * 1e3 + 0.3, 40.0, 22.5, 0.1)
    clear_mm = st.number_input("Clearance cells → casing (mm)", 0.5, 20.0, 2.0, 0.5)
    wall_mm = st.number_input("Casing wall thickness (mm)", 0.5, 10.0, 3.0, 0.5)
    filler_name = st.selectbox("Material between cells", list(FILLERS))
    casing_name = st.selectbox("Casing material", list(CASINGS))
    pack_depth_mm = st.number_input("Pack depth along cell axis (mm)", 50.0, 800.0, 300.0, 10.0,
                                    help="Used only for the natural-convection length scale of the horizontal faces.")

with sb.expander("3 · Custom operating case", expanded=False):
    st.caption("Used when the main-page scenario is **Custom**.")
    mode = st.radio("Battery state", ["Riding (discharge)", "Charging"], horizontal=True)
    veh = Vehicle(n_series=int(n_s), n_parallel=int(n_p))
    if mode.startswith("Riding"):
        how = st.radio("Load from", ["Steady cruise speed (road-load model)", "Pack current directly"])
        speed = st.slider("Vehicle speed (km/h)", 0, 90, 60)
        grade = st.slider("Road grade (%)", 0, 15, 0)
        veh.mass_rider = st.slider("Rider + payload (kg)", 50, 175, 75)
        P_batt_peak = veh.p_peak / veh.eta_drive
        if how.startswith("Steady"):
            P_batt = min(cruise_power(speed, veh, grade), P_batt_peak)
            I_pack = pack_current(P_batt, veh)
        else:
            I_pack = st.slider("Pack current (A)", 0, 160, 60)
            P_batt = I_pack * veh.v_pack_nom
    else:
        speed = 0
        grade = 0
        I_pack = -float(st.slider("Charging current (A)", 1, 60, 15,
                                  help="Home charger for 2.9 kWh ≈ 0–80 % in 3 h ≈ 15 A. "
                                       f"Datasheet max charge 3.395 A/cell × {int(n_p)}P = {3.395*n_p:.0f} A."))
        P_batt = I_pack * veh.v_pack_nom
    I_cell = I_pack / n_p

with sb.expander("4 · Environment and boundaries", expanded=False):
    uf = st.slider("Underbody airflow factor", 0.0, 1.0, 0.5, 0.05,
                   help="Fraction of vehicle speed that reaches the pack underside (assumption).")
    eps = st.slider("Casing emissivity", 0.05, 0.95, 0.30, 0.05,
                    help="0.2–0.3 bare cast aluminium, ~0.9 painted/powder-coated.")
    kinds = {"Vehicle airflow (forced + natural)": "airflow", "Natural convection": "natural",
             "Fixed temperature": "fixed", "Insulated": "insulated"}
    kinds_lr = {k: v for k, v in kinds.items() if v != "airflow"}
    side_cfg = {}
    for side, default, opts in [("bottom", 0, kinds), ("top", 1, kinds), ("left", 0, kinds_lr), ("right", 0, kinds_lr)]:
        choice = st.selectbox(f"{side.capitalize()} face", list(opts), index=default, key=f"bc_{side}")
        k = opts[choice]
        Tfix = 30.0
        if k == "fixed":
            Tfix = st.number_input(f"{side.capitalize()} face temperature (°C)", 0.0, 80.0, 30.0, 1.0, key=f"tf_{side}")
        side_cfg[side] = (k, Tfix)

with sb.expander("5 · Numerics", expanded=False):
    h_mm_mesh = st.select_slider("Target element size (mm)", [3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8], value=1.5)
    t_end_min = st.slider("Max transient duration (min)", 5, 300, 240, 5)
    dt_s = st.select_slider("Time step (s)", [2, 5, 10, 20, 30, 60], value=10)

# =============================================================================
# Main page: question, scenario, ambient
# =============================================================================
st.title("Ather 450X battery: how hot does it get?")
st.caption("2-D finite element model of the real 21700-cell pack (14S12P, 168 cells) · "
           "RV College of Engineering, Mechanical Engineering · Experiential Learning Phase II")
SCEN = {"City · 40 km/h": ("ride", 40), "Highway · 60 km/h": ("ride", 60), "Fast · 75 km/h": ("ride", 75),
        "Aggressive riding": ("spirited", 60), "Home charging · 15 A": ("charge", None),
        "Custom (sidebar)": ("custom", None)}
cS, cA = st.columns([3, 1])
scen = cS.segmented_control("Scenario", list(SCEN), default="Highway · 60 km/h", key="scen") or "Highway · 60 km/h"
T_amb = cA.slider("Ambient temperature (°C)", 15, 48, 35)
scen_kind, scen_speed = SCEN[scen]
if scen_kind in ("ride", "spirited"):
    speed, grade = scen_speed, 0
    P_batt = min(cruise_power(speed, veh, 0), veh.p_peak / veh.eta_drive)
    I_pack = pack_current(P_batt, veh)
elif scen_kind == "charge":
    speed, grade = 0, 0
    I_pack = -15.0
    P_batt = I_pack * veh.v_pack_nom
I_cell = I_pack / n_p
if scen_kind == "spirited":
    # steady "held forever" bound uses the RMS current of the burst cycle (same average I^2 R heat)
    _Ip = veh.p_peak / veh.eta_drive / veh.v_pack_nom
    I_cell = math.sqrt((_Ip ** 2 * 15 + I_pack ** 2 * 45) / 60) / n_p
ride_kind = scen_kind if scen_kind != "custom" else ("ride" if I_cell >= 0 else "charge")

# =============================================================================
# Build and solve (cached)
# =============================================================================
geo = PackGeometry(cell_d=cell.diameter, pitch=pitch_mm * 1e-3, rows=int(rows), cols=int(cols),
                   layout=layout, clearance=clear_mm * 1e-3, wall=wall_mm * 1e-3)
env = Environment(T_amb_C=float(T_amb), speed_kmh=float(speed), underbody_factor=uf, emissivity=eps,
                  pack_depth=pack_depth_mm * 1e-3)
fil = FILLERS[filler_name]
cas = CASINGS[casing_name]
dUdT = dUdT_mV * 1e-3
Ea = Ea_kJ * 1e3
limit = cell.t_max_discharge_C if I_cell >= 0 else cell.t_max_charge_C

try:
    geo.validate()
except ValueError as e:
    st.error(str(e))
    st.stop()

key = json.dumps(dict(geo=geo.__dict__, env=env.__dict__, cell=cell.__dict__, fil=fil.__dict__, cas=cas.__dict__,
                      sides=side_cfg, I_cell=I_cell, dUdT=dUdT, Ea=Ea, h=h_mm_mesh), sort_keys=True, default=str)


def build_problem(geo, env, cell, fil, cas, sides, dUdT, Ea, h_mm):
    m = mesh_pack(geo, h_mm * 1e-3)
    bcs = {s: SideBC(k, tf) for s, (k, tf) in sides.items()}
    return PackProblem(
        mesh=m, k_cell_r=cell.k_radial, k_cell_t=cell.k_axial, k_filler=fil.k, k_casing=cas.k,
        rhoc_cell=cell.density * cell.cp, rhoc_filler=fil.rho * fil.cp, rhoc_casing=cas.rho * cas.cp,
        cell_area_true=math.pi * cell.radius ** 2, cell_length=cell.height, side_bc=bcs,
        h_func=lambda s, bc, Ts: side_h(s, bc, Ts, env, geo.W, geo.H),
        q_cell_func=lambda Tc, Ic: cell_heat(Ic, Tc, cell, Ea=Ea, dUdT=dUdT),
        T_inf=env.T_amb_C)


@st.cache_resource(show_spinner=False, max_entries=6)
def get_problem(key):
    return build_problem(geo, env, cell, fil, cas, side_cfg, dUdT, Ea, h_mm_mesh)


@st.cache_data(show_spinner=False, max_entries=12)
def steady(key):
    prob = get_problem(key)
    t0 = time.time()
    T, hs, Q, hist = prob.solve_steady(I_cell)
    eb = prob.energy_balance(T, hs, Q)
    return dict(T=T, hs=hs, Q=Q, hist=hist, eb=eb, cm=prob.cell_means(T), cx=prob.cell_max(T),
                secs=time.time() - t0)


with st.spinner("Meshing and solving steady state..."):
    prob = get_problem(key)
    res = steady(key)
mesh = prob.mesh
T = res["T"]

# =============================================================================
# Headline: a realistic full ride (transient until the battery is empty)
# =============================================================================
Tmax = float(T.max())
cm, cx = res["cm"], res["cx"]
spread = float(cm.max() - cm.min())
Qc = res["Q"]
ok_T = Tmax <= limit
ok_dT = spread <= 5.0
USABLE_WH = 2600.0                       # Ather 450X 2.9 kWh pack: 2.6 kWh usable [4]
I_peak = veh.p_peak / veh.eta_drive / veh.v_pack_nom


@st.cache_data(show_spinner=False, max_entries=24)
def ride(key, kind):
    if kind == "spirited":
        I_avg = (I_peak * 15 + I_pack * 45) / 60
        f = lambda t: (I_peak if (t % 60) < 15 else I_pack) / n_p
        dt = 5.0
    else:
        I_avg = abs(I_pack)
        f = lambda t: I_cell
        dt = 20.0
    if kind == "charge":   # 0 -> 80 % of the 2.9 kWh installed capacity
        t_end = 0.8 * 2900.0 / max(I_avg * veh.v_pack_nom, 1e-6) * 3600
    else:                  # until the 2.6 kWh usable energy is spent
        t_end = USABLE_WH / max(I_avg * veh.v_pack_nom, 1e-6) * 3600
    t_end = min(t_end, 4 * 3600)
    pr = get_problem(key)
    r = pr.solve_transient(f, t_end, dt, env.T_amb_C, store_every=10 ** 9)
    return dict(t=r["t"], Tmax=r["Tmax"], T=r["T"], t_end=t_end, cm=pr.cell_means(r["T"]), I_avg=I_avg)


with st.spinner("Simulating the ride..."):
    rr = ride(key, ride_kind)
ride_peak = float(rr["Tmax"].max())
ride_min = rr["t_end"] / 60
ride_lim = cell.t_max_charge_C if ride_kind == "charge" else cell.t_max_discharge_C
cross = np.where(rr["Tmax"] > ride_lim)[0]
what = {"ride": f"A full ride at {speed} km/h, until the battery is empty ({ride_min:.0f} min),",
        "spirited": f"Aggressive riding (full-throttle bursts every minute) until the battery is empty ({ride_min:.0f} min)",
        "charge": f"Charging at {abs(I_pack):.0f} A from 0 to 80 % ({ride_min:.0f} min)"}[ride_kind]
if len(cross) == 0:
    st.success(f"**Within limits.** {what} peaks at **{ride_peak:.1f} °C**, below the {ride_lim:.0f} °C "
               f"{'charge' if ride_kind == 'charge' else 'discharge'} limit of the cells (ambient {T_amb} °C).")
else:
    st.error(f"**Over the limit.** {what} crosses the {ride_lim:.0f} °C cell limit after "
             f"**{rr['t'][cross[0]] / 60:.0f} min** and reaches {ride_peak:.1f} °C. The battery management system "
             f"would have to reduce power.")

ride_spread = float(rr["cm"].max() - rr["cm"].min())
k1, k2, k3 = st.columns(3)
k1.metric("Peak during the charge" if ride_kind == "charge" else "Peak during the ride", f"{ride_peak:.1f} °C", f"{ride_peak - ride_lim:+.1f} K vs {ride_lim:.0f} °C limit",
          delta_color="inverse")
k2.metric("Worst case if held forever", f"{Tmax:.1f} °C",
          help="Steady state: the temperature the pack would settle at if this load never ended. "
               "A real ride ends first, so this is an upper bound.")
k3.metric("Hottest vs coolest cell", f"{ride_spread:.1f} K",
          help="Temperature difference between the hottest and coolest cell. A common design target is 5 K or less, "
               "because hotter cells age faster.")

f1, f2 = st.columns([2.2, 1])
with f1:
    st.plotly_chart(plots.field_figure(mesh, rr["T"], f"Pack cross-section at the end ({ride_min:.0f} min), °C"),
                    width="stretch")
with f2:
    import plotly.graph_objects as go
    figt = go.Figure(go.Scatter(x=rr["t"] / 60, y=rr["Tmax"], line=dict(color="#d1495b", width=2.5),
                                name="Hottest point"))
    figt.add_hline(y=ride_lim, line_dash="dash", line_color="#8a1c2b", annotation_text=f"limit {ride_lim:.0f} °C")
    figt.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10), title="Hottest point over time",
                       xaxis_title="Time (min)", yaxis_title="°C", showlegend=False)
    st.plotly_chart(figt, width="stretch")

with st.expander("What am I looking at?"):
    st.markdown(
        "- The picture is a slice through the battery pack, cut across the cells. Each circle is one real "
        "21700 cell (21 mm across); the frame around them is the aluminium casing.\n"
        "- Current flowing through each cell heats it. That heat has to cross the plastic cell holder and the casing, "
        "then leave into the air. Moving air under the scooter cools the bottom face more than the others.\n"
        "- The answer comes from a finite element model: the slice is split into about 40,000 small triangles and "
        "the heat equation is solved on each. The tabs below show the worst case, the full equations, proof that the "
        "solver is correct, and files to repeat the analysis in ANSYS.")

tabs = st.tabs(["Worst case (steady)", "Explore over time", "Geometry & mesh", "Equations",
                "Verification", "ANSYS", "Sources & assumptions"])

# =============================================================================
# Steady state
# =============================================================================
with tabs[0]:
    st.caption("Steady state = the temperature the pack settles at if this load continues forever. It bounds the "
               "real ride shown above from above.")
    if ok_T and ok_dT:
        st.success(f"Even held forever, the peak ({Tmax:.1f} °C) stays below the {limit:.0f} °C limit and the cell "
                   f"spread ({spread:.1f} K) below 5 K.")
    else:
        msgs = []
        if not ok_T:
            msgs.append(f"the peak would reach {Tmax:.1f} °C (limit {limit:.0f} °C)")
        if not ok_dT:
            msgs.append(f"the cell spread would reach {spread:.1f} K (target 5 K)")
        st.info("If this load were held forever, " + " and ".join(msgs) + ". A real ride ends before this; see the "
                "result at the top of the page.")
    st.plotly_chart(plots.field_figure(mesh, T, "Steady-state temperature field (°C)"), width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(plots.cell_bar(cm, cx, limit), width="stretch")
    with c2:
        st.markdown("**Boundary heat transfer (converged)**")
        hrows = []
        for s, d in res["hs"].items():
            kind = side_cfg[s][0]
            hrows.append({"Face": s, "Type": kind, "h_forced": d["h_forced"], "h_natural": d["h_natural"],
                          "h_conv": d["h_conv"], "h_rad": d["h_rad"], "h_total (W/m²K)": d["h"],
                          "Heat out (W/m)": res["eb"]["by_side"][s]})
        st.dataframe(pd.DataFrame(hrows).style.format(precision=2), hide_index=True, width="stretch")
        eb = res["eb"]
        st.markdown(f"**Global energy balance** (per metre depth): generated **{eb['q_gen']:.2f} W/m**, "
                    f"leaving **{eb['q_out']:.2f} W/m**, residual **{eb['residual_pct']:.1e} %**.")
        st.markdown(f"Picard iterations: **{len(res['hist'])}** (h(T) and R(T) updated each pass) · "
                    f"mesh {mesh.n_nodes:,} nodes / {mesh.n_elems:,} elements · solve {res['secs']:.2f} s")
        Ric = cell_resistance(cell.dcir, cm, cell.t_ref_C, Ea)
        st.markdown(f"Per-cell heat: irreversible I²R = {I_cell**2 * Ric.mean():.3f} W, reversible "
                    f"−I·T·dU/dT = {-I_cell * (cm.mean()+273.15) * dUdT:.3f} W.")
    st.info("**Reading this result.** Steady state assumes the load is held indefinitely, so it bounds what a "
            "long continuous ride can reach. The 2-D section also omits heat flow along the cell axis to the "
            "end plates and busbars, which makes it conservative. Use the Transient tab for an actual ride or "
            "charge duration.")

# =============================================================================
# Transient
# =============================================================================
with tabs[1]:
    st.markdown("Backward-Euler time integration of **M dT/dt + K T = f(t)** from a uniform initial temperature. "
                "h(T) and R(T) are lagged one step, and the system is refactorised whenever any face h drifts "
                "by more than 3 %.")
    cA, cB, cC = st.columns(3)
    prof = cA.selectbox("Load profile", ["Constant (current operating case)",
                                         "Spirited ride: full-throttle bursts + cruise",
                                         "Charging at constant current"])
    T0 = cB.number_input("Initial pack temperature (°C)", 0.0, 60.0, float(T_amb), 1.0)
    burst = cC.number_input("Full-throttle burst (s)", 5, 120, 15)
    period = cC.number_input("Cycle period (s)", 20, 600, 60)
    I_peak = veh.p_peak / veh.eta_drive / veh.v_pack_nom
    if prof.startswith("Spirited"):
        st.caption(f"Peak: {I_peak:.0f} A ({veh.p_peak/1000:.1f} kW motor / η {veh.eta_drive}) for {burst} s "
                   f"in every {period} s, otherwise the current operating case ({I_pack:.0f} A).")
    if prof.startswith("Charging") and I_cell >= 0:
        st.caption("Uses a 15 A charge (switch the sidebar to *Charging* to set the current and the charge limit).")

    def I_of_t(t, prof=prof):
        if prof.startswith("Constant"):
            return I_cell
        if prof.startswith("Spirited"):
            return (I_peak if (t % period) < burst else I_pack) / n_p
        return (I_pack if I_cell < 0 else -15.0) / n_p

    usable_Wh = st.number_input("Usable pack energy (Wh)", 500.0, 6000.0, 2600.0, 50.0,
                                help="Ather 450X 2.9 kWh pack: 2.6 kWh usable [4]. The run stops when it is used up.")
    if prof.startswith("Spirited"):
        I_avg = (I_peak * burst + I_pack * (period - burst)) / period
    elif prof.startswith("Charging"):
        I_avg = abs(I_pack) if I_cell < 0 else 15.0
    else:
        I_avg = abs(I_pack)
    t_empty_min = usable_Wh / max(I_avg * veh.v_pack_nom, 1e-6) * 60
    if prof.startswith("Charging"):
        t_empty_min *= 0.8
    t_run_min = min(float(t_end_min), t_empty_min)
    st.caption(f"Average current {I_avg:.1f} A → {'0→80 % charge' if prof.startswith('Charging') else 'pack empty'} "
               f"after {t_empty_min:.0f} min. Simulated duration: **{t_run_min:.0f} min** "
               f"(the smaller of this and the sidebar duration).")
    run = st.button("Run transient", type="primary")
    if run:
        with st.spinner("Time stepping..."):
            t0 = time.time()
            tr_res = prob.solve_transient(I_of_t, t_run_min * 60.0, float(dt_s), T0,
                                          store_every=max(1, int(60 / dt_s)))
            tr_res["I"] = tr_res["I"] * n_p
            tr_res["secs"] = time.time() - t0
        st.session_state["tr"] = (key, prof, tr_res)
    if "tr" in st.session_state and st.session_state["tr"][0] == key:
        _, prof_used, tr_res = st.session_state["tr"]
        lim_tr = cell.t_max_charge_C if prof_used.startswith("Charging") else cell.t_max_discharge_C
        st.plotly_chart(plots.transient_figure(tr_res, lim_tr), width="stretch")
        k1, k2, k3 = st.columns(3)
        k1.metric("Peak at end", f"{tr_res['Tmax'][-1]:.1f} °C")
        k2.metric("Max over run", f"{tr_res['Tmax'].max():.1f} °C")
        cross = np.where(tr_res["Tmax"] > lim_tr)[0]
        k3.metric("Time to limit", f"{tr_res['t'][cross[0]]/60:.1f} min" if len(cross) else "not reached")
        snaps = tr_res["snaps"]
        idx = st.slider("Snapshot", 0, len(snaps) - 1, len(snaps) - 1,
                        format="%d", help="Stored every minute of simulated time.")
        ts, Ts = snaps[idx]
        st.plotly_chart(plots.field_figure(mesh, Ts, f"Temperature at t = {ts/60:.1f} min",
                                           zmin=float(T0), zmax=float(tr_res['Tmax'].max())),
                        width="stretch")
        st.caption(f"{len(tr_res['t'])-1} steps in {tr_res['secs']:.1f} s.")
    else:
        st.info("Set the profile and press **Run transient**.")

# =============================================================================
# Geometry & mesh
# =============================================================================
with tabs[2]:
    g1, g2 = st.columns([3, 2])
    with g1:
        st.plotly_chart(plots.mesh_figure(mesh), width="stretch")
    with g2:
        st.markdown("**Dimensions used**")
        dims = pd.DataFrame([
            ["Cell diameter", f"{cell.diameter*1e3:.2f} mm", "LG M50T datasheet [6]"],
            ["Cell height", f"{cell.height*1e3:.2f} mm", "LG M50/M50T datasheet [5][6]"],
            ["Cell mass", f"{cell.mass*1e3:.1f} g", "LG M50T datasheet [6]"],
            ["Cell density (derived)", f"{cell.density:.0f} kg/m³", "mass / volume"],
            ["Cells per pack", f"{n_s*n_p} ({n_s}S{n_p}P)", "Community-reported [4]"],
            ["Pack nominal voltage", "51.1 V", "Ather spec sheet [3]"],
            ["Holder pitch", f"{geo.pitch*1e3:.1f} mm", "Assumed (typical 21700 holder)"],
            ["Cell gap", f"{(geo.pitch-cell.diameter)*1e3:.2f} mm", "pitch − diameter"],
            ["Casing wall", f"{geo.wall*1e3:.1f} mm", "Assumed (Al casing confirmed [3])"],
            ["Section W × H", f"{geo.W*1e3:.1f} × {geo.H*1e3:.1f} mm", "Derived from the above"],
            ["Cells in section", f"{geo.n_cells}", f"{rows} × {cols}, {layout}"],
        ], columns=["Quantity", "Value", "Source"])
        st.dataframe(dims, hide_index=True, width="stretch")
        st.markdown(f"Mesh: **{mesh.n_nodes:,} nodes, {mesh.n_elems:,} linear triangles**, min angle 30° "
                    "(Triangle [15]). Cell boundaries are area-preserving polygons, so heat input and "
                    "heat capacity are not under-counted.")
        st.caption("Ather does not publish the internal pack layout. The default 3 × 14 section with "
                   "4 such slices along the cell axis gives 168 cells. Change rows and columns to explore "
                   "other layouts.")

# =============================================================================
# Governing equations
# =============================================================================
with tabs[3]:
    st.subheader("1. Energy equation (2-D, per unit depth)")
    st.latex(r"\rho c_p \frac{\partial T}{\partial t} = \nabla\cdot\left(\mathbf{K}\,\nabla T\right) + \dot q''' \qquad \text{in } \Omega")
    st.markdown("Each region has its own properties. The casing and filler are isotropic "
                "($\\mathbf K = k\\mathbf I$). Inside a cell the spiral-wound electrode stack makes the conductivity "
                "**cylindrically orthotropic**: low across the layers (radial) and high along them "
                "(tangential, ≈ the measured axial value) [7][8][11]:")
    st.latex(r"\mathbf K_\text{cell} = k_r\,\mathbf e_r\mathbf e_r^{\mathsf T} + k_\theta\,\mathbf e_\theta\mathbf e_\theta^{\mathsf T},\qquad \mathbf e_r=\frac{\mathbf x-\mathbf x_c}{\lVert\mathbf x-\mathbf x_c\rVert}")
    st.subheader("2. Heat generation (Bernardi et al. [1])")
    st.latex(r"Q_\text{cell} = I_c^2\,R(T)\;-\;I_c\,T\,\frac{\partial U_\text{OCV}}{\partial T},\qquad I_c=\frac{I_\text{pack}}{N_p},\qquad \dot q''' = \frac{Q_\text{cell}}{\pi r^2 L}")
    st.latex(r"R(T) = R_\text{ref}\,\exp\!\left[\frac{E_a}{R_g}\left(\frac1T-\frac1{T_\text{ref}}\right)\right]")
    st.markdown("$I_c>0$ on discharge. The first term is irreversible (ohmic plus polarisation, lumped in the "
                "DCIR); the second is the reversible entropic heat [2]. Mixing enthalpy and phase change are "
                "neglected (Bernardi's simplified form).")
    st.subheader("3. Boundary conditions")
    st.latex(r"-\mathbf n\cdot\mathbf K\nabla T = h_\text{eff}\,(T - T_\infty),\qquad h_\text{eff}=h_\text{conv}+h_\text{rad}")
    st.latex(r"h_\text{rad} = \varepsilon\sigma\,(T_s+T_\infty)(T_s^2+T_\infty^2)")
    st.markdown("Fixed-temperature (Dirichlet) and insulated (zero-flux) faces are also available. "
                "Forced convection on the bottom/top faces uses the average flat-plate correlations [9]:")
    st.latex(r"\overline{Nu}_L = 0.664\,Re_L^{1/2}Pr^{1/3}\;(Re_L<5\times10^5),\qquad \overline{Nu}_L=(0.037Re_L^{4/5}-871)Pr^{1/3}")
    st.markdown("Natural convection uses Churchill–Chu for the vertical faces [10] and the hot-plate "
                "correlations for horizontal faces, with $L=A_s/P$ [9]:")
    st.latex(r"\overline{Nu}_L=\left\{0.825+\frac{0.387Ra_L^{1/6}}{[1+(0.492/Pr)^{9/16}]^{8/27}}\right\}^2,\quad \overline{Nu}_\text{up}=0.54Ra^{1/4}\;/\;0.15Ra^{1/3},\quad \overline{Nu}_\text{down}=0.52Ra^{1/5}")
    st.markdown("Mixed convection blends the two as $Nu^n = Nu_F^n + Nu_N^n$, with $n=3.5$ for horizontal plates "
                "[9, Eq. 9.64]. Air properties are taken at the film temperature (Incropera Table A.4). Because "
                "$h$ depends on the surface temperature, the steady problem is nonlinear and is solved by "
                "Picard iteration with under-relaxation.")
    st.subheader("4. Galerkin weak form and element matrices [12][13]")
    st.latex(r"\int_\Omega \rho c_p\,w\,\dot T\,d\Omega + \int_\Omega \nabla w\cdot\mathbf K\nabla T\,d\Omega + \int_{\Gamma_R} h\,wT\,d\Gamma = \int_\Omega w\,\dot q'''\,d\Omega + \int_{\Gamma_R} h\,wT_\infty\,d\Gamma")
    st.markdown("With linear triangles $T\\approx\\sum N_iT_i$, $N_i=(a_i+b_ix+c_iy)/2A_e$:")
    st.latex(r"\mathbf B=\frac1{2A_e}\begin{bmatrix}b_1&b_2&b_3\\c_1&c_2&c_3\end{bmatrix},\quad b_i=y_j-y_k,\;c_i=x_k-x_j")
    st.latex(r"\mathbf K_e = A_e\,\mathbf B^{\mathsf T}\mathbf K\,\mathbf B,\qquad \mathbf M_e=\frac{\rho c_pA_e}{12}\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},\qquad \mathbf f_e=\frac{\dot q'''A_e}{3}\begin{Bmatrix}1\\1\\1\end{Bmatrix}")
    st.latex(r"\mathbf H_e=\frac{hL_e}{6}\begin{bmatrix}2&1\\1&2\end{bmatrix},\qquad \mathbf r_e=\frac{hT_\infty L_e}{2}\begin{Bmatrix}1\\1\end{Bmatrix}")
    st.markdown("Global assembly gives $\\mathbf M\\dot{\\mathbf T}+(\\mathbf K+\\mathbf H)\\mathbf T=\\mathbf f+\\mathbf r$. "
                "Dirichlet nodes are eliminated and the sparse system is solved by LU factorisation. "
                "Time integration uses the θ-method with θ = 1 (backward Euler, unconditionally stable):")
    st.latex(r"\left(\frac{\mathbf M}{\Delta t}+\theta\mathbf K\right)\mathbf T^{n+1}=\left(\frac{\mathbf M}{\Delta t}-(1-\theta)\mathbf K\right)\mathbf T^{n}+\mathbf f^{n+\theta}")
    st.subheader("5. Load from vehicle speed (road-load model)")
    st.latex(r"P_\text{batt}=\frac{\left[m g\,(C_{rr}\cos\alpha+\sin\alpha)+\tfrac12\rho_a C_dA\,v^2\right]v}{\eta_\text{drive}},\qquad I_\text{pack}=\frac{P_\text{batt}}{V_\text{nom}}")
    st.markdown(f"Kerb mass 108 kg and V_nom 51.1 V are from Ather [3]. C_rr = {veh.crr}, C_dA = {veh.CdA} m² and "
                f"η = {veh.eta_drive} are assumptions. Battery power is capped at the 6.4 kW peak motor power / η.")

# =============================================================================
# Verification
# =============================================================================
with tabs[4]:
    st.markdown("Verification answers *is the equation solved correctly?* Three independent checks follow.")
    st.subheader("A. Analytical benchmark: single orthotropic cell")
    st.latex(r"T(r)=T_\infty+\frac{\dot q R}{2h}+\frac{\dot q\,(R^2-r^2)}{4k_r}")
    st.caption("The cell has k_r = 0.9 and k_θ = 24.2 W/m·K. Only k_r appears in the exact solution, so a wrong "
               "tensor rotation would show up immediately.")

    @st.cache_data(show_spinner=False)
    def _v1():
        rows_, best = ver.single_cell_case()
        m, Tn, Tex, r = best
        return rows_, Tn, Tex, r

    rows_v1, Tn1, Tex1, r1 = _v1()
    v1, v2 = st.columns(2)
    v1.dataframe(pd.DataFrame(rows_v1).style.format(precision=4), hide_index=True, width="stretch")
    v2.plotly_chart(plots.radial_profile_figure(r1, Tn1, Tex1), width="stretch")
    st.plotly_chart(plots.convergence_figure([r["h_mm"] for r in rows_v1], [r["L2_err_K"] for r in rows_v1],
                                             "Element size h (mm)", "L2 error (K)", ref_slope=2),
                    width="stretch")

    st.subheader("B. Method of manufactured solutions: observed order of accuracy")
    st.latex(r"T_\text{ex}=\sin\pi x\,\sin\pi y,\qquad \dot q'''=2\pi^2k\,\sin\pi x\,\sin\pi y,\qquad T=0\text{ on }\partial\Omega")

    @st.cache_data(show_spinner=False)
    def _v2():
        return ver.mms_case()

    mms = pd.DataFrame(_v2())
    st.dataframe(mms.style.format(precision=4), hide_index=True, width="stretch")
    st.caption("Linear triangles should converge at second order in the L2 norm. The observed order approaching "
               "2 confirms the assembly, quadrature and Dirichlet treatment.")

    st.subheader("C. Mesh convergence of this pack model (Grid Convergence Index)")
    st.markdown("The current case is re-solved on three meshes. The peak temperature is extrapolated and the "
                "fine-grid GCI is reported following Celik et al. [14] (safety factor 1.25).")
    if st.button("Run mesh study (3 meshes)"):
        hs_list = [h_mm_mesh * 2.0, h_mm_mesh * 1.4, h_mm_mesh]
        out = []
        with st.spinner("Solving three meshes..."):
            for hh in hs_list:
                pb = build_problem(geo, env, cell, fil, cas, side_cfg, dUdT, Ea, hh)
                Tt, hs_, Q_, _ = pb.solve_steady(I_cell)
                out.append(dict(h_mm=hh, elements=pb.mesh.n_elems, nodes=pb.mesh.n_nodes, T_max=float(Tt.max()),
                                hottest_cell_mean=float(pb.cell_means(Tt).max())))
        df = pd.DataFrame(out)
        st.dataframe(df.style.format(precision=4), hide_index=True, width="stretch")
        g = ver.gci(df["elements"], df["T_max"])
        st.markdown(f"Observed order **p = {g['p']:.2f}** ({g['note']}), extrapolated peak "
                    f"**{g['phi_ext']:.3f} °C**, fine-grid **GCI = {g['gci_fine_pct']:.3f} %** "
                    f"(r₂₁ = {g['r21']:.2f}, r₃₂ = {g['r32']:.2f}).")
        st.session_state["gci"] = (key, df, g)
    st.subheader("D. Energy conservation")
    st.markdown(f"For the current steady case, the heat leaving all faces matches the heat generated within "
                f"**{abs(res['eb']['residual_pct']):.1e} %**.")

# =============================================================================
# ANSYS cross-check
# =============================================================================
with tabs[5]:
    st.markdown("This app does not run ANSYS. It generates ready-to-run **ANSYS Mechanical APDL** input files for "
                "the case currently on screen. Run them in ANSYS and upload the result file back here for a "
                "node-by-node comparison.")
    mat = dict(k_r=cell.k_radial, k_t=cell.k_axial, rho_cell=cell.density, cp_cell=cell.cp,
               k_fill=fil.k, rho_fill=fil.rho, cp_fill=fil.cp, k_case=cas.k, rho_case=cas.rho, cp_case=cas.cp)
    q_cell = Qc / (math.pi * cell.radius ** 2 * cell.height)
    kinds_now = {s: v[0] for s, v in side_cfg.items()}
    tfix_now = {s: v[1] for s, v in side_cfg.items()}
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**A · Same mesh (code verification)**")
        st.caption("This app's nodes and elements written to APDL as PLANE55. Expect agreement to round-off.")
        st.download_button("Download same_mesh.inp",
                           same_mesh_apdl(mesh, mat, res["hs"], kinds_now, tfix_now, q_cell, env.T_amb_C),
                           file_name="ather450x_same_mesh.inp", mime="text/plain")
    with a2:
        st.markdown("**B · ANSYS mesh (solution verification)**")
        st.caption("Geometry rebuilt from primitives and meshed by ANSYS with 8-node PLANE77. Expect agreement "
                   "within the GCI.")
        st.download_button("Download ansys_mesh.inp",
                           geometry_apdl(geo, mat, res["hs"], kinds_now, tfix_now, q_cell, env.T_amb_C,
                                         esize=h_mm_mesh * 1e-3),
                           file_name="ather450x_ansys_mesh.inp", mime="text/plain")
    with a3:
        st.markdown("**C · Geometry for Workbench**")
        st.caption("DXF in mm (layers: CASING_OUTER, CASING_INNER, CELLS) for SpaceClaim / DesignModeler.")
        st.download_button("Download section.dxf", geometry_dxf(geo), file_name="ather450x_section.dxf",
                           mime="application/dxf")

    with st.expander("How to run in ANSYS Mechanical APDL", expanded=True):
        st.markdown(f"""
1. Open **Mechanical APDL** (Ansys Student works; the same-mesh file has {mesh.n_nodes:,} nodes, which is
   below the 128k-node student limit).
2. **File → Read Input from…** and pick the `.inp` file. The script builds the model, solves it, plots the
   temperature field and writes `ansys_nodal_*.csv` to the working directory.
3. Compare the peak temperature shown in the message box with this app's **{Tmax:.3f} °C**.
4. Upload the CSV below for a full nodal comparison.
5. Check that ANSYS honours the cell orthotropy: `/PSYMB,ESYS,1` then `EPLOT` should show radial triads
   inside every cell. Material X (KXX = k_r) is radial in each cell's local cylindrical system
   (`LOCAL,{100}+i,1`).
""")
    with st.expander("How to build the same model in ANSYS Workbench (Steady-State Thermal)"):
        st.markdown(f"""
1. **Geometry**: in SpaceClaim open `section.dxf`, then *Fill* the regions to make surfaces: the casing ring,
   the filler, and {geo.n_cells} cell discs. Use *Share Topology* so the mesh is conformal. In
   **Geometry properties** set *Analysis Type = 2D*.
2. **Engineering Data**:
   - Cell: *Orthotropic Thermal Conductivity* X = {cell.k_radial} (radial), Y = {cell.k_axial}
     (tangential), Z = {cell.k_axial} W/m·K; density {cell.density:.0f} kg/m³; c_p {cell.cp:.0f} J/kg·K.
   - Filler: k = {fil.k}, ρ = {fil.rho:.0f}, c_p = {fil.cp:.0f}.
   - Casing: k = {cas.k}, ρ = {cas.rho:.0f}, c_p = {cas.cp:.0f}.
3. **Coordinate systems**: insert one *Cylindrical* coordinate system per cell at its centre (centres are in
   the APDL file, `LOCAL` lines) and assign it to that body (*Body → Coordinate System*).
4. **Mesh**: element size {h_mm_mesh} mm (or finer for a convergence check).
5. **Loads**: *Internal Heat Generation* on each cell body ({q_cell.mean():.4g} W/m³ for uniform R).
   *Convection* on the outer edges, with h per face from the table below and ambient {env.T_amb_C:.1f} °C.
   Fixed temperature where selected.
6. Solve and compare the maximum temperature with **{Tmax:.3f} °C**.
""")
        st.dataframe(pd.DataFrame([{"Face": s, "Type": kinds_now[s], "h_eff (W/m²K)": round(d["h"], 4),
                                    "T_fixed (°C)": tfix_now[s] if kinds_now[s] == "fixed" else None}
                                   for s, d in res["hs"].items()]), hide_index=True)
        st.caption("These h values are the app's converged effective coefficients (convection plus "
                   "linearised radiation), so both solvers see identical boundary conditions.")

    st.markdown("**Upload ANSYS results** (`ansys_nodal_T_*.csv`)")
    up = st.file_uploader("CSV written by either APDL script", type=["csv", "txt"])
    if up is not None:
        data = parse_ansys_csv(up.getvalue().decode("utf-8", errors="ignore"))
        if len(data) == 0:
            st.error("No node,x,y,T rows found in that file.")
        else:
            T_app = interpolate(mesh, T, data[:, :2])
            ok = ~np.isnan(T_app)
            diff = data[ok, 2] - T_app[ok]
            cA, cB, cC, cD = st.columns(4)
            cA.metric("ANSYS peak", f"{data[:, 2].max():.3f} °C")
            cB.metric("App peak", f"{Tmax:.3f} °C", f"{Tmax - data[:, 2].max():+.3f} K")
            cC.metric("Max |ΔT| at nodes", f"{np.abs(diff).max():.3e} K")
            cD.metric("RMS ΔT", f"{np.sqrt(np.mean(diff**2)):.3e} K")
            import plotly.graph_objects as go
            fig = go.Figure(go.Scattergl(x=T_app[ok], y=data[ok, 2], mode="markers",
                                         marker=dict(size=3, color=diff, colorscale="RdBu", colorbar=dict(title="ΔT"))))
            lo, hi = float(np.nanmin(T_app)), float(np.nanmax(T_app))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#888"),
                                     showlegend=False))
            fig.update_layout(height=380, xaxis_title="App T (°C)", yaxis_title="ANSYS T (°C)",
                              margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
            st.plotly_chart(fig, width="stretch")

# =============================================================================
# Assumptions & references
# =============================================================================
with tabs[6]:
    st.subheader("Data sources: what is real and what is assumed")
    st.markdown(f"""
**From published sources**
- 450X pack: 2.9 kWh / 3.7 kWh (now 3.5 kWh) installed, **51.1 V nominal**, IP67, **aluminium-alloy casing**,
  6.4 kW peak motor, 90 km/h, 108 kg kerb (Ather spec sheet [3]).
- **21700 cylindrical cells**, which replaced the 18650 cells of the original Ather 450 [4].
- Cell **Ø21.10 × 70.15 mm, 68 g, 5.0 Ah, 3.63 V, DCIR 30 ± 6 mΩ** (LG INR21700-M50/M50T [5][6]).
  168 × 18.2 Wh = 3.06 kWh, consistent with the 2.9 kWh rating.
- Cell thermal properties **k_r ≈ 0.9, k_axial ≈ 24.2 W/m·K, c_p ≈ 885 J/kg·K**, measured on a 21700 cell [7].
  These fall inside the ranges of the Steinhardt meta-analysis [8].

**Assumed (Ather does not publish these)**
- Cell maker: LG M50T is used as a representative high-energy 21700 NMC cell. Ather has not disclosed its supplier.
- **14S12P** configuration (community-reported [4]); the section layout (rows × columns), holder pitch, clearance,
  wall thickness, filler material and casing alloy.
- Underbody airflow factor, emissivity, rolling resistance, CdA and drivetrain efficiency.

**Modelling simplifications**
- **2-D section normal to the cell axes**: axial conduction to the end caps, busbars and holder plates is
  neglected. These are extra cooling paths, so the predicted temperatures are conservative.
- The thin steel can (~0.2 mm) and the cell-to-holder contact resistance are not resolved. Cells are assumed in
  perfect contact with the filler.
- Lumped DCIR heat (no electrochemical model); uniform current sharing between parallel cells.
- Natural convection inside the pack is neglected when *Still air* is chosen as filler (conduction only).
- Fixed-temperature faces assume an ideal isothermal sink with no contact resistance (best case).
- No thermal runaway, ageing or SOC dependence of R or dU/dT.
""")
    st.subheader("References")
    st.markdown(ref_markdown())
