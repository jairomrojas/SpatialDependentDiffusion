# Scientific and numerical validation report

Audit date: 30 July 2026  
Code/artifact schema: `chem-swimmer 0.2.0`, density schema 2

## Executive conclusion

The manuscript's forward equation, angular closure, effective diffusivity,
chemotactic mobility, signs, and Case 1-8 mapping have been re-derived and
agree with the implementation. No sign or missing-factor error was found in

\[
\gamma_0=D_{R0}+\lambda_0(1-a_1),
\qquad
D_{\rm eff}=D_{T0}+\frac{v_0^2}{2\gamma_0},
\]

\[
\kappa=\frac{v^S}{2}-D_T^c
-\frac{v_0}{2\gamma_0}
\left[v^c+D_R^S+(1-a_1)\lambda^S\right].
\]

The apparent disagreement involving Cases 2, 4, and 5 was not caused by an
error in this formula. It came from comparing a bulk, long-time,
first-order closure with a bounded microscopic model and from using
nonnegative image norms whose sampling-noise floor differs between cases.
The default sliding wall creates an angular boundary layer that the scalar
closed equation no longer contains.

The most important correction to the earlier interpretation is Case 5. Its
bulk mobility is zero at retained order, but its measured full-disk
microscopic response is linear in \(D_T^S\), because the sliding wall has a
nonzero baseline polarity layer. It should not be described simply as a
quadratic microscopic effect.

## Independent equation audit

### Itô forward operator

For

\[
d\mathbf x=v\mathbf e\,dt+\sqrt{2D_T}\,d\mathbf W_T,\qquad
d\theta=\sqrt{2D_R}\,dW_R,
\]

with left-endpoint Itô coefficients, the continuous forward terms are

\[
-\nabla\cdot(v\mathbf eP)
+\nabla^2(D_TP)+\partial_\theta^2(D_RP).
\]

The code evaluates every coefficient at the pre-step state, so the
multiplicative-noise convention agrees with the manuscript. Replacing
\(\nabla^2(D_TP)\) by \(\nabla\cdot(D_T\nabla P)\) would be a different model;
the code does not make that replacement.

### Tumble operator

The turn law is achiral and rotation invariant, with
\(\mathbb E[\cos\Delta\theta]=a_1\). The first angular harmonic therefore
relaxes through \(\lambda_0(1-a_1)\), giving the stated \(\gamma_0\).
The particular wrapped-normal sampler also fixes higher harmonics, for
example its second harmonic. Those higher moments are a microscopic kernel
choice, not determined by \(a_1\) or by the closed density theory.

### Exact scalar moment before closure

Angular integration gives the translational contribution

\[
\nabla^2\left[
D_{T0}\rho+D_T^c(c-c_0)\rho+
D_T^S\,\mathbf p\cdot\nabla c
\right].
\]

Thus Cases 2 and 4 act only through angular moments. Case 5 acts through
\(\mathbf p\cdot\nabla c\). All three vanish as first-order scalar forcing
about an isotropic, uniform *bulk* reference state. That cancellation does
not hold automatically in a kinetic boundary layer where
\(\mathbf p_{\rm wall}^{(0)}\ne0\).

### Macroscopic finite volume

The Scharfetter-Gummel face rates have the correct sign. For neighboring
cells \(L,R\), their rate ratio is

\[
\frac{q_{L\to R}}{q_{R\to L}}
=\exp\left[\frac{\kappa(c_R-c_L)}{D_{\rm eff}}\right],
\]

so the discrete equilibrium is exactly
\(\rho_i\propto\exp(\kappa c_i/D_{\rm eff})\). The explicit timestep uses the
maximum total outgoing rate, which makes every update coefficient
nonnegative. Unit probability is conserved to the tested floating-point
tolerance.

## Focused audit of Cases 2, 3, 4, and 5

The standard output uses a preselected signed observable,
\(\Delta\langle c\rangle\), and paired standard errors. At \(t=0.5\):

| Case | final signed response | paired \(z\) | absolute size / Case 3 |
|---:|---:|---:|---:|
| 2 | \(+0.002241\) | \(2.54\) | \(0.353\) |
| 3 | \(+0.006354\) | \(25.48\) | \(1\) |
| 4 | \(-0.001306\) | \(-0.54\) | \(0.205\) |
| 5 | \(-0.001649\) | \(-12.54\) | \(0.259\) |

The familywise 95-percent threshold for the eight comparisons is \(2.734\).
Therefore Cases 2 and 4 are not resolved by this preselected test, Case 5 is
resolved, and none is larger than Case 3 on the signed common-scale
observable.

The \(M=16384\) sign-reversal study in
`output/data/response_scaling_audit.csv` isolates
odd and even response components. Under the sliding wall:

| Case | odd at \(|\alpha|=0.5\) | odd at \(|\alpha|=1\) | finding |
|---:|---:|---:|---|
| 2 | \(0.001215\pm0.000448\) | \(0.001775\pm0.000741\) | weak/borderline and not cleanly scaled |
| 3 | \(0.003096\pm0.000119\) | \(0.006209\pm0.000233\) | linear |
| 4 | \(-0.002143\pm0.001210\) | \(-0.000973\pm0.001240\) | unresolved |
| 5 | \(-0.000768\pm0.000066\) | \(-0.001559\pm0.000130\) | linear |

Case 5's even component is consistent with zero. Cases 3 and 5 both double
when \(|\alpha|\) doubles. This supports a linear sliding-wall contribution
for Case 5 and rejects the explanation that its observed full-disk signal is
mainly the quadratic bulk correction.

Specular reflection greatly increases paired variance because a small change
in a collision changes the reflected heading and rapidly decorrelates the
two paths. Those results are useful sensitivity evidence but are not a
high-power null test at the same \(M\).

## Numerical refinement evidence

The saved `numerical_convergence_audit.csv` uses \(M=4096\) at the automatic
timestep and refinements by factors two and four:

| Case | \(dt\) | \(dt/2\) | \(dt/4\) |
|---:|---:|---:|---:|
| 2 | \(0.00245\pm0.00176\) | \(-0.00102\pm0.00172\) | \(-0.00234\pm0.00174\) |
| 3 | \(0.00528\pm0.00048\) | \(0.00609\pm0.00051\) | \(0.00552\pm0.00052\) |
| 4 | \(0.00364\pm0.00483\) | \(-0.00199\pm0.00480\) | \(0.00448\pm0.00483\) |
| 5 | \(-0.00161\pm0.00026\) | \(-0.00169\pm0.00027\) | \(-0.00112\pm0.00026\) |

The innovations are reproducible within each refinement but are not
Brownian-bridge coupled between different \(dt\), so cross-row differences
must be compared with their combined Monte Carlo errors. Cases 2 and 4 are
not timestep-stable signals. Cases 3 and 5 retain their sign and compatible
magnitude.

For the macroscopic equilibrium, grids \(81^2,121^2,161^2,241^2,321^2\)
give Case 3 exposure values differing by less than \(6\times10^{-8}\). The
stronger Case 7 changes from \(0.0634266\) to \(0.0634070\), about 0.03
percent. Conservation, positivity, and exact discrete equilibrium are also
covered by regression tests.

## Corrections made during this audit

1. The example custom field used \(\cos(4\theta)\) without the required
   \(r^4\) factor. It was not continuous at the origin despite being described
   as smooth. It now uses the smooth Cartesian polynomial
   \(x^4-6x^2y^2+y^4=r^4\cos4\theta\), with a regression test at the origin.
2. The Python density fallback and fused Numba engine now use the same
   per-particle initial-state and innovation draw order. Histograms agree
   exactly; reduced moments agree to floating-point roundoff.
3. Output archives now embed the complete TOML input, its SHA-256 digest,
   schema/software versions, and runtime versions.
4. The automatic \(M\) calculation is now described accurately as a
   uniform-kernel occupancy floor, not a guarantee that the KDE has converged.
5. Diagnostics consistently use the manuscript symbol \(\kappa\), clearly
   state that the TOML is already dimensionless, and describe Brownian
   timestep limits as RMS targets rather than pathwise bounds.
6. A reproducible odd/even response-scaling and wall-law audit was added.

## Remaining model assumptions and limits

- The microscopic collision law is additional model input; `docs/theory.pdf`
  supplies the closed no-flux condition but does not uniquely select sliding,
  specular, or diffuse particle-wall behavior.
- Euler-Maruyama and finite-step reflection converge only under refinement.
  The automatic timestep is a conservative starting target, not an error
  certificate.
- At most one tumble is represented per step. At the configured upper bound
  \(\lambda dt=0.05\), a frozen Poisson process has probability about
  \(0.00121\) of two or more events. Reducing `max_lambda_dt` tests this
  approximation.
- The KDE boundary correction reproduces a uniform disk exactly after
  normalization but can retain boundary bias for a nonuniform target.
  Quantitative density morphology requires \(M\)- and bandwidth-refinement.
- The digitized circular finite-volume boundary converges geometrically as
  the grid is refined; exact conservation at one grid is not a substitute for
  a grid-convergence study.
- For a user-defined field, a sampled grid plus a declared characteristic
  length cannot rigorously exclude unresolved extrema. The user is
  responsible for supplying a smooth field, an accurate gradient, and a
  defensible smallest length.

## Primary and official sources consulted

- Wagner, Hagan, and Baskaran, *Steady states of active Brownian particles
  interacting with boundaries*, Phys. Rev. E 105, 024610 (2022):
  <https://arxiv.org/abs/2109.06353>.
- Duzgun and Selinger, *Active Brownian particles near straight or curved
  walls: pressure and boundary layers*, Phys. Rev. E 97, 032606 (2018):
  <https://arxiv.org/abs/1712.09627>.
- Gobet, *Euler schemes and half-space approximation for the simulation of
  diffusion in a domain*, ESAIM Probability and Statistics 5 (2001):
  <https://www.esaim-ps.org/articles/ps/pdf/2001/01/psVol5-13.pdf>.
- Scharfetter and Gummel, *Large-signal analysis of a silicon Read diode
  oscillator*, IEEE Trans. Electron Devices 16 (1969), DOI
  <https://doi.org/10.1109/T-ED.1969.16566>.
- Marron and Ruppert, *Transformations to reduce boundary bias in kernel
  density estimation*, JRSS B 56 (1994):
  <https://doi.org/10.1111/j.2517-6161.1994.tb02006.x>.
- NumPy official guidance on `SeedSequence` and parallel reproducibility:
  <https://numpy.org/doc/stable/reference/random/parallel.html>.
- Numba official supported-NumPy documentation, including independent
  compiled random state:
  <https://numba.readthedocs.io/en/stable/reference/numpysupported.html>.
- Python Packaging User Guide on `pyproject.toml` and the `src` layout:
  <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>.
- pytest official integration practices:
  <https://docs.pytest.org/en/latest/explanation/goodpractices.html>.
