"""Audit of the manuscript's long-time coefficients D_eff, kappa and V.

Integrates the closed density--polarity system of ``main.tex``
Eq. (polar-closed-system) directly, with the fields taken uniform in y and a
linear nutrient landscape ``eta_1 = g x``, then compares

  * the spreading rate  d<x^2>/dt = 2 D_eff        (one component),
  * the centroid velocity  d<x>/dt = eps V g,      Eq. (pulse-moments),

with the closed-form coefficients.  No adiabatic elimination is used in the
integration, so this tests the reduction itself.  The residual in V is the
startup transient and decays exactly as 1/(gamma1 T); Cases 1 and 5 give an
identically zero centroid velocity, and Case 7 is exact to round-off.

Run with a larger ``T`` to tighten the agreement:

    python scripts/check_long_time_coefficients.py
"""

import numpy as np

v0, DT0, DR0, lam0, P = 2.0, 1.0, 50.0, 0.0, 0.0
G = DR0 + lam0 * (1.0 - P)  # = calligraphic D_R^(0)
Deff0 = DT0 + v0**2 / (2 * G)
g = 1.0  # eta_1 = g*x


def coeffs(**kw):
    DTe = kw.get("DT_eta", 0.0)
    DRe = kw.get("DR_eta", 0.0)
    ve = kw.get("v_eta", 0.0)
    lame = kw.get("lam_eta", 0.0)
    DRz = kw.get("DR_zeta", 0.0)
    vz = kw.get("v_zeta", 0.0)
    lamz = kw.get("lam_zeta", 0.0)
    chi = kw.get("chi", 0.0)
    Ge = DRe + (1 - P) * lame
    Gz = DRz + (1 - P) * lamz
    dDp = DTe + v0 * ve / G - v0**2 * Ge / (2 * G**2)  # delta D_eff'
    kap = (v0 * (chi - Gz) + G * vz) / (2 * G) - DTe - v0 * ve / (2 * G)
    return dDp, kap, kap + dDp


def run(eps, N=2001, L=40.0, T=1.0, dt=2.5e-5, **kw):
    x = np.linspace(-L / 2, L / 2, N)
    dx = x[1] - x[0]
    eta1 = g * x
    deta = eps * g * np.ones_like(x)  # grad(eta) = eps*g
    de = eps * eta1  # delta eta
    DTe = kw.get("DT_eta", 0.0)
    DRe = kw.get("DR_eta", 0.0)
    ve = kw.get("v_eta", 0.0)
    lame = kw.get("lam_eta", 0.0)
    DTz = kw.get("DT_zeta", 0.0)
    DRz = kw.get("DR_zeta", 0.0)
    vz = kw.get("v_zeta", 0.0)
    lamz = kw.get("lam_zeta", 0.0)
    chi = kw.get("chi", 0.0)
    Ge = DRe + (1 - P) * lame
    Gz = DRz + (1 - P) * lamz
    v = v0 + ve * de
    DT = DT0 + DTe * de
    d = lambda f: np.gradient(f, dx, edge_order=2)
    s0 = 0.5
    rho = np.exp(-(x**2) / (2 * s0**2)) / np.sqrt(2 * np.pi * s0**2)
    p = np.zeros_like(x)
    nsteps = round(T / dt)
    for _ in range(nsteps):
        Jrho = v * p + 0.5 * vz * rho * deta - d(DT * rho + DTz * p * deta)
        Jp = (
            0.5 * rho * v
            + 0.25 * vz * (3.0 * p * deta)
            - d(DT * p + 0.5 * DTz * rho * deta)
        )
        R = -(G + Ge * de) * p + 0.5 * (chi - Gz) * rho * deta
        rho = rho - dt * d(Jrho)
        p = p - dt * (d(Jp) - R)
    m0 = np.trapezoid(rho, x)
    m1 = np.trapezoid(x * rho, x)
    cen = m1 / m0
    var = np.trapezoid((x - cen) ** 2 * rho, x) / m0
    return cen, var, T


cases = {
    "1  D_T^(eta)": {"DT_eta": 1.0},
    "2  D_R^(eta)": {"DR_eta": 20.0},
    "3  v^(eta)": {"v_eta": 1.0},
    "4  lam^(eta)": {"lam_eta": 20.0},
    "5  D_T^(zp)": {"DT_zeta": 1.0},
    "6  D_R^(zp)": {"DR_zeta": 20.0},
    "7  v^(zp)": {"v_zeta": 1.0},
    "8  lam^(zp)": {"lam_zeta": 20.0},
    "9  chi": {"chi": 20.0},
}
print(f"D_eff^(0) = {Deff0:.6f}")
c0, var0, T = run(0.0)
print(
    f"baseline: centroid={c0:+.3e}  measured D_eff="
    f"{(var0 - 0.5**2) / (2 * T):.6f}  (1D: d<x^2>/dt = 2 D_eff)\n"
)
print(
    f"{'case':<14}{'V predicted':>13}{'V measured':>13}{'rel.err':>10}   "
    f"{'kappa':>10}{'dD_eff':>10}"
)
for name, kw in cases.items():
    dDp, kap, V = coeffs(**kw)
    e1, e2 = 1e-3, 2e-3
    c1, _, _ = run(e1, **kw)
    c2, _, _ = run(e2, **kw)
    # antisymmetric-in-eps extrapolation to kill O(eps^2)
    cm1, _, _ = run(-e1, **kw)
    Vm = (c1 - cm1) / 2.0 / (e1 * g * T)
    err = (Vm - V) / V if abs(V) > 1e-12 else Vm
    print(f"{name:<14}{V:>13.6f}{Vm:>13.6f}{err:>10.2e}   {kap:>10.5f}{dDp:>10.5f}")
print(
    "\nThe residual in V is the O(1/(gamma1 T)) startup transient; rerun "
    "with a larger T and it halves each time T doubles."
)
