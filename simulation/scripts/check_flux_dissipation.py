"""Audit of the finite-volume face-flux dissipation (moment.py).

A one-dimensional reduction of the closed (rho,p) system, integrated with the
same face-flux construction the package uses, so the numerical diffusivity can
be measured directly against theory.

  Rusanov   adds a numerical diffusivity a*dx/2 on every grid.  At the paper
            configuration that is ~0.33, a third of D_T0, and it carries
            v_eta and v_zeta, so it does NOT cancel in the case-minus-baseline
            response: it is what contaminated Cases 3 and 7.
  hybrid    adds max(0, a/2 - D_T/dx), which is exactly zero once the cell
            Peclet number a*dx/D_T <= 2 and still keeps the scheme monotone
            when it is not.  This is what moment.py now uses.

Expected output: "central" and "hybrid" agree and are grid independent; the
"rusanov" diffusivity exceeds them by exactly v0*dx/2, halving with dx.

Rusanov adds a numerical diffusivity S*dx/2 unconditionally.  The physical
diffusion already supplies D_T.  The standard hybrid coefficient
    a = max(0, S/2 - D_T/dx)
keeps the scheme monotone at large cell Peclet number and reduces to plain
central differencing once D_T/dx >= S/2, i.e. Pe_cell = S*dx/D_T <= 2.
"""

import numpy as np

v0, DT0, DR0, lam0, P = 26.67, 1.0, 178.0, 254.0, 0.30
G = DR0 + lam0 * (1 - P)


def run(dx, mode, T=5e-4, dt=2.0e-8, v_eta=0.0, g=0.0, L=4.0):
    n = round(L / dx)
    x = (np.arange(n) - n // 2) * dx
    s0 = 0.10
    rho = np.exp(-(x**2) / (2 * s0**2)) / np.sqrt(2 * np.pi * s0**2)
    p = np.zeros_like(x)
    v = v0 + v_eta * g * x
    va, vb = v[:-1], v[1:]
    S = np.maximum(np.abs(va), np.abs(vb))
    if mode == "rusanov":
        a = 0.5 * S
    elif mode == "central":
        a = 0.0 * S
    else:
        a = np.maximum(0.0, 0.5 * S - DT0 / dx)
    for _ in range(round(T / dt)):
        Frho = (
            0.5 * (va * p[:-1] + vb * p[1:])
            - a * (rho[1:] - rho[:-1])
            - DT0 * (rho[1:] - rho[:-1]) / dx
        )
        Fp = (
            0.25 * (va * rho[:-1] + vb * rho[1:])
            - a * (p[1:] - p[:-1])
            - DT0 * (p[1:] - p[:-1]) / dx
        )
        drho = np.zeros_like(rho)
        dp = np.zeros_like(p)
        drho[:-1] -= Frho / dx
        drho[1:] += Frho / dx
        dp[:-1] -= Fp / dx
        dp[1:] += Fp / dx
        dp -= G * p
        rho = rho + dt * drho
        p = p + dt * dp
    m0 = rho.sum() * dx
    cen = (x * rho).sum() * dx / m0
    return cen, ((x - cen) ** 2 * rho).sum() * dx / m0, T, float(rho.min())


print(
    f"{'grid':>6} {'Pe_cell':>8} "
    + "".join(f"{m:>22}" for m in ("rusanov", "central", "hybrid"))
)
for g_ in (21, 41, 81, 161, 321):
    dx = 2.0 / g_
    row = f"{g_:>6} {v0 * dx / DT0:>8.2f} "
    for mode in ("rusanov", "central", "hybrid"):
        _, var, T, mn = run(dx, mode)
        row += f"  D={(var - 0.01) / (2 * T):>8.5f} min={mn:>8.1e}"
    print(row)

print("\nCase 3 spurious centroid velocity (v_eta=4), per unit eps*g")
print(f"{'grid':>6} {'rusanov':>12} {'central':>12} {'hybrid':>12}")
for g_ in (81, 161, 321):
    dx = 2.0 / g_
    eps = 1e-3
    row = f"{g_:>6} "
    for mode in ("rusanov", "central", "hybrid"):
        c1, _, T, _ = run(dx, mode, v_eta=4.0, g=eps)
        c0, _, _, _ = run(dx, mode, v_eta=0.0, g=eps)
        row += f"{(c1 - c0) / (eps * T):>12.5f}"
    print(row)
