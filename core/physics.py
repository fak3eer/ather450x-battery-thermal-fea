"""
Physical models: cell data, material properties, heat generation,
convection/radiation correlations and a scooter road-load model.

All quantities in SI units unless the name says otherwise.
Reference numbers [n] point to core/references.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import numpy as np

SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann constant, W/m2K4
R_GAS = 8.314462618         # J/mol-K
G = 9.80665                 # m/s2


# --------------------------------------------------------------------------
# Cell data
# --------------------------------------------------------------------------
@dataclass
class CellSpec:
    name: str
    diameter: float          # m
    height: float            # m
    mass: float              # kg
    capacity_Ah: float       # nominal
    v_nom: float             # V
    dcir: float              # ohm, DC internal resistance at T_ref
    t_ref_C: float           # deg C at which dcir is quoted
    k_radial: float          # W/m-K (across the spiral wound layers)
    k_axial: float           # W/m-K (along the layers; used as in-plane tangential k)
    cp: float                # J/kg-K
    dUdT: float              # V/K entropic coefficient (sign: dOCV/dT)
    t_max_discharge_C: float
    t_max_charge_C: float
    source: str = ""

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.height

    @property
    def density(self) -> float:
        return self.mass / self.volume

    @property
    def energy_Wh(self) -> float:
        return self.capacity_Ah * self.v_nom


# LG INR21700-M50/M50T. Geometry, mass, capacity, voltage and DCIR from the
# LG product specification [5][6]; thermal properties measured on a 21700
# NMC cell [7]; operating limits from the M50T datasheet [6].
LG_M50T = CellSpec(
    name="LG INR21700-M50T (21700 NMC811/Gr-SiOx)",
    diameter=21.10e-3,
    height=70.15e-3,
    mass=68.0e-3,
    capacity_Ah=5.0,
    v_nom=3.63,
    dcir=30e-3,
    t_ref_C=25.0,
    k_radial=0.90,
    k_axial=24.2,
    cp=885.0,
    dUdT=0.0,
    t_max_discharge_C=60.0,
    t_max_charge_C=45.0,
    source="LG spec sheet [5][6]; thermal properties [7]",
)


# --------------------------------------------------------------------------
# Materials for the non-cell regions
# --------------------------------------------------------------------------
@dataclass
class Material:
    name: str
    k: float        # W/m-K (isotropic)
    rho: float      # kg/m3
    cp: float       # J/kg-K
    note: str = ""


FILLERS = {
    "Cell holder plastic (PC/ABS)": Material("PC/ABS cell holder", 0.20, 1150.0, 1250.0,
                                             "Typical polycarbonate/ABS datasheet values"),
    "Thermally conductive potting": Material("Thermal potting compound", 0.80, 1600.0, 1000.0,
                                             "Typical filled PU/silicone potting (0.5-1.5 W/m-K)"),
    "Still air": Material("Air (no internal convection)", 0.0263, 1.16, 1007.0,
                          "Incropera Table A.4 at 300 K [9]"),
}

CASINGS = {
    "Aluminium alloy, die-cast (ADC12-type)": Material("Die-cast Al alloy", 96.0, 2740.0, 963.0,
                                                       "Typical ADC12/A383 handbook value"),
    "Aluminium alloy 6061-T6 (extruded/sheet)": Material("Al 6061-T6", 167.0, 2700.0, 896.0,
                                                         "ASM handbook value"),
}


# --------------------------------------------------------------------------
# Heat generation (Bernardi et al. 1985 [1], simplified form)
# --------------------------------------------------------------------------
def cell_resistance(R_ref: float, T_C: np.ndarray | float, T_ref_C: float, Ea: float) -> np.ndarray:
    """Arrhenius temperature dependence of DCIR: R = R_ref exp[Ea/Rg (1/T - 1/T_ref)].
    Ea = 0 gives a constant resistance."""
    T = np.asarray(T_C, dtype=float) + 273.15
    Tr = T_ref_C + 273.15
    return R_ref * np.exp(Ea / R_GAS * (1.0 / T - 1.0 / Tr))


def cell_heat(I_cell: float, T_C: np.ndarray | float, cell: CellSpec, Ea: float = 0.0,
              dUdT: float | None = None) -> np.ndarray:
    """Heat generation per cell [W].

    Simplified Bernardi energy balance [1] (no enthalpy of mixing, no phase change):
        Q = I^2 R(T)  -  I * T * dU/dT
    I > 0 for discharge, I < 0 for charge. T in kelvin in the entropic term.
    """
    dudt = cell.dUdT if dUdT is None else dUdT
    T_K = np.asarray(T_C, dtype=float) + 273.15
    R = cell_resistance(cell.dcir, T_C, cell.t_ref_C, Ea)
    return I_cell ** 2 * R - I_cell * T_K * dudt


# --------------------------------------------------------------------------
# Air properties (Incropera Table A.4 [9]) and convection correlations
# --------------------------------------------------------------------------
_AIR_T = np.array([250.0, 300.0, 350.0, 400.0])
_AIR_NU = np.array([11.44, 15.89, 20.92, 26.41]) * 1e-6
_AIR_K = np.array([22.3, 26.3, 30.0, 33.8]) * 1e-3
_AIR_PR = np.array([0.720, 0.707, 0.700, 0.690])


def air_props(T_film_C: float):
    T = T_film_C + 273.15
    return (float(np.interp(T, _AIR_T, _AIR_NU)),
            float(np.interp(T, _AIR_T, _AIR_K)),
            float(np.interp(T, _AIR_T, _AIR_PR)))


def nu_forced_flat_plate(Re: float, Pr: float) -> float:
    """Average Nusselt number, flat plate parallel flow (Incropera Eqs 7.30, 7.38 [9])."""
    if Re <= 0:
        return 0.0
    if Re < 5e5:
        return 0.664 * Re ** 0.5 * Pr ** (1 / 3)
    return (0.037 * Re ** 0.8 - 871.0) * Pr ** (1 / 3)


def nu_natural(Ra: float, Pr: float, orientation: str) -> float:
    """Average Nusselt number for natural convection (Incropera Ch. 9 [9]).
    orientation: 'vertical'  -> Churchill & Chu, full range (Eq 9.26) [10]
                 'up'        -> upper surface of hot plate (Eqs 9.30, 9.31)
                 'down'      -> lower surface of hot plate (Eq 9.32)
    """
    Ra = max(Ra, 1e-9)
    if orientation == "vertical":
        return (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    if orientation == "up":
        return 0.54 * Ra ** 0.25 if Ra < 1e7 else 0.15 * Ra ** (1 / 3)
    if orientation == "down":
        return 0.52 * Ra ** 0.2
    raise ValueError(orientation)


@dataclass
class SideBC:
    kind: str                 # 'airflow' | 'natural' | 'fixed' | 'insulated'
    T_fixed_C: float = 30.0


@dataclass
class Environment:
    T_amb_C: float = 32.0
    speed_kmh: float = 40.0
    underbody_factor: float = 0.5     # fraction of vehicle speed seen by the pack underside
    emissivity: float = 0.3
    pack_depth: float = 0.30          # m, pack extent along the cell axis (normal to the 2-D section)


def side_h(side: str, bc: SideBC, T_surf_C: float, env: Environment,
           W: float, H: float) -> dict:
    """Effective heat transfer coefficient (convection + linearised radiation)
    for one outer face of the casing. Returns a breakdown dict.

    side: 'bottom' | 'top' | 'left' | 'right'
    W, H: outer width and height of the 2-D section (m).
    """
    T_inf = env.T_amb_C
    Tf = 0.5 * (T_surf_C + T_inf)
    nu, k, Pr = air_props(Tf)
    beta = 1.0 / (Tf + 273.15)
    dT = max(abs(T_surf_C - T_inf), 1e-3)
    out = dict(h_forced=0.0, h_natural=0.0, h_conv=0.0, h_rad=0.0, h=0.0, Re=0.0, Ra=0.0)
    if bc.kind in ("fixed", "insulated"):
        return out

    # --- natural convection
    if side in ("left", "right"):
        L = H
        orient = "vertical"
    else:
        # horizontal plate W x pack_depth, L = A/P (Incropera Eq 9.29)
        L = (W * env.pack_depth) / (2 * (W + env.pack_depth))
        hot = T_surf_C >= T_inf
        if side == "top":
            orient = "up" if hot else "down"
        else:
            orient = "down" if hot else "up"
    Ra = G * beta * dT * L ** 3 / (nu * (nu / Pr))   # nu/Pr = alpha
    Nu_N = nu_natural(Ra, Pr, orient)

    # --- forced convection (only bottom/top faces see the vehicle air stream)
    Nu_F = 0.0
    Re = 0.0
    if bc.kind == "airflow" and side in ("bottom", "top"):
        U = env.speed_kmh / 3.6 * env.underbody_factor
        Lf = W                      # flow runs along the section width (vehicle length)
        Re = U * Lf / nu
        Nu_F = nu_forced_flat_plate(Re, Pr)
        h_F = Nu_F * k / Lf
        h_N = Nu_N * k / L
        # mixed convection blending, Incropera Eq 9.64 with n = 3.5 for horizontal plates [9]
        n = 3.5
        h_conv = (h_F ** n + h_N ** n) ** (1 / n)
        out.update(h_forced=h_F, h_natural=h_N, h_conv=h_conv, Re=Re, Ra=Ra)
    else:
        h_N = Nu_N * k / L
        out.update(h_natural=h_N, h_conv=h_N, Ra=Ra)

    # --- linearised radiation to surroundings at T_inf
    Ts = T_surf_C + 273.15
    Ti = T_inf + 273.15
    h_rad = env.emissivity * SIGMA_SB * (Ts + Ti) * (Ts ** 2 + Ti ** 2)
    out["h_rad"] = h_rad
    out["h"] = out["h_conv"] + h_rad
    return out


# --------------------------------------------------------------------------
# Road-load model: vehicle speed -> battery power -> pack current
# --------------------------------------------------------------------------
@dataclass
class Vehicle:
    mass_kerb: float = 108.0      # kg, Ather 450X 2.9 kWh [3]
    mass_rider: float = 75.0
    crr: float = 0.015
    CdA: float = 0.50             # m2 (assumed: scooter + rider)
    rho_air: float = 1.15
    eta_drive: float = 0.85       # battery-to-wheel efficiency (assumed)
    p_peak: float = 6400.0        # W, motor peak power [3]
    v_pack_nom: float = 51.1      # V, nominal pack voltage [3]
    n_series: int = 14            # [4]
    n_parallel: int = 12          # [4]


def cruise_power(v_kmh: float, veh: Vehicle, grade_pct: float = 0.0) -> float:
    """Battery power [W] for steady cruise at v on a grade."""
    v = v_kmh / 3.6
    m = veh.mass_kerb + veh.mass_rider
    theta = math.atan(grade_pct / 100.0)
    F = m * G * (veh.crr * math.cos(theta) + math.sin(theta)) + 0.5 * veh.rho_air * veh.CdA * v ** 2
    return max(F * v / veh.eta_drive, 0.0)


def pack_current(P_batt: float, veh: Vehicle) -> float:
    return P_batt / veh.v_pack_nom


def to_dict(obj) -> dict:
    return asdict(obj)
