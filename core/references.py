"""Numbered references used throughout the code and the app."""

REFERENCES = [
    # 1
    ("Bernardi, D., Pawlikowski, E., Newman, J. (1985). A General Energy Balance for Battery Systems. "
     "Journal of The Electrochemical Society 132(1), 5-12.",
     "https://doi.org/10.1149/1.2113792"),
    # 2
    ("Viswanathan, V.V., Choi, D., Wang, D., et al. (2010). Effect of entropy change of lithium "
     "intercalation in cathodes and anodes on Li-ion battery thermal management. "
     "Journal of Power Sources 195(11), 3720-3729.",
     "https://doi.org/10.1016/j.jpowsour.2009.11.103"),
    # 3
    ("Ather Energy (2024). Ather 450 Series brochure / spec sheet: 2.9 kWh and 3.7 kWh installed capacity, "
     "51.1 V nominal pack voltage, IP67, aluminium-alloy pack casing, 6.4 kW peak motor, 90 km/h, 108 kg kerb.",
     "https://media.atherenergy.com/Ather-450-Spec-Sheet-Oct-2024.pdf"),
    # 4
    ("Ather Community Forum. 450X battery discussions: 21700 cells replace the 18650 cells of the Ather 450; "
     "pack reported as 168 cells, 14S12P (community-reported, not an official Ather figure).",
     "https://forum.atherenergy.com/t/how-is-the-battery-of-the-ather-450x-different-from-that-of-ather-450/5454"),
    # 5
    ("LG Chem. Product Specification, Rechargeable Lithium Ion Battery INR21700 M50 18.2 Wh "
     "(height <= 70.15 mm, 68.0 g, 4.85 Ah min / 5.0 Ah nominal, 3.63 V). Summary incl. DCIR 30 +/- 6 mOhm "
     "(0.5C, 30 s) at batterydesign.net.",
     "https://www.batterydesign.net/lg-21700-m50/"),
    # 6
    ("LG Chem. INR21700-M50T datasheet (diameter 21.10 mm max, height 70.15 mm, 68.0 +/- 1.0 g; "
     "charge 0-45 C, discharge -20-60 C).",
     "https://www.keeppower.com.cn/products_detail.php?id=577"),
    # 7
    ("Measurement of thermophysical parameters and thermal modeling of 21700 cylindrical battery. "
     "Journal of Energy Storage (2023): cp ~ 885 J/kg-K, radial k ~ 0.9 W/m-K, axial k ~ 24.2 W/m-K.",
     "https://www.sciencedirect.com/science/article/abs/pii/S2352152X23007351"),
    # 8
    ("Steinhardt, M., et al. (2021/2022). Low-effort determination of heat capacity "
     "and thermal conductivity for cylindrical 18650 and 21700 lithium-ion cells; and Meta-analysis of experimental "
     "results for heat capacity and thermal conductivity in lithium-ion batteries. Journal of Energy Storage. "
     "(Cylindrical cells: radial k 0.15-1.4 W/m-K, axial k 21-40 W/m-K.)",
     "https://www.researchgate.net/publication/357877737"),
    # 9
    ("Bergman, T.L., Lavine, A.S., Incropera, F.P., DeWitt, D.P. (2011). Fundamentals of Heat and Mass "
     "Transfer, 7th ed. Wiley. (Table A.4 air properties; Eqs 7.30, 7.38 flat plate; Eqs 9.26-9.32 natural "
     "convection; Eq 9.64 mixed convection.)", ""),
    # 10
    ("Churchill, S.W., Chu, H.H.S. (1975). Correlating equations for laminar and turbulent free convection "
     "from a vertical plate. International Journal of Heat and Mass Transfer 18(11), 1323-1329.",
     "https://doi.org/10.1016/0017-9310(75)90243-4"),
    # 11
    ("Drake, S.J., Wetz, D.A., Ostanek, J.K., et al. (2014). Measurement of anisotropic thermophysical properties "
     "of cylindrical Li-ion cells. Journal of Power Sources 252, 298-304.",
     "https://doi.org/10.1016/j.jpowsour.2013.11.107"),
    # 12
    ("Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z. (2013). The Finite Element Method: Its Basis and Fundamentals, "
     "7th ed. Butterworth-Heinemann. (Galerkin weak form, linear triangle, consistent capacity matrix.)", ""),
    # 13
    ("Lewis, R.W., Nithiarasu, P., Seetharamu, K.N. (2004). Fundamentals of the Finite Element Method for Heat "
     "and Fluid Flow. Wiley. (Thermal FEM, Robin boundary matrices, theta-method time stepping.)", ""),
    # 14
    ("Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., Coleman, H., Raad, P.E. (2008). Procedure for "
     "Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications. "
     "ASME Journal of Fluids Engineering 130(7), 078001. (Grid Convergence Index.)",
     "https://doi.org/10.1115/1.2960953"),
    # 15
    ("Shewchuk, J.R. (1996). Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator. "
     "Applied Computational Geometry, LNCS 1148, 203-222.",
     "https://www.cs.cmu.edu/~quake/triangle.html"),
    # 16
    ("Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering. Hermosa. "
     "(Method of manufactured solutions, observed order of accuracy.)", ""),
    # 17
    ("Ansys Inc. Mechanical APDL Element Reference: PLANE55 (2-D thermal solid) and PLANE77 (2-D 8-node "
     "thermal solid); Command Reference: ESYS, LOCAL, SF/SFL, BFE, *VGET, *VWRITE.", ""),
]


def ref_markdown() -> str:
    lines = []
    for i, (txt, url) in enumerate(REFERENCES, 1):
        lines.append(f"{i}. {txt}" + (f" [{url}]({url})" if url else ""))
    return "\n".join(lines)
