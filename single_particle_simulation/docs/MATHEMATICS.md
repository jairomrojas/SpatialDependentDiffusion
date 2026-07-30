# Mathematics-to-code map

## State and signals

The state is position `r = (x, y)` and heading `theta`. The heading vector and
alignment signal are

```text
e(theta) = (cos(theta), sin(theta))
S(r, theta) = e(theta) dot grad(c(r)).
```

Every field supplies both `c(r)` and `grad(c(r))`. Its `baseline` is the
reference `c0`, so `delta_c = c - c0`.

## Radius and diffusion-time nondimensionalization

The input scales define

```text
x = r_physical/R,                       |x| <= 1,
t = t_physical/t_D,        t_D = R^2/D_T0.
```

From this point onward unstarred symbols denote dimensionless variables:

```text
v      = v_physical R/D_T0,
D_T    = D_T_physical/D_T0,
D_R    = D_R_physical R^2/D_T0,
lambda = lambda_physical R^2/D_T0.
```

These converted values—not the dimensional measurements—are what the TOML
configuration accepts. In particular, `domain.radius = 1`,
`kinetics.D_T0 = 1`, and `simulation.duration` is the final dimensionless
time directly.

Consequently the computational update is

```text
Delta x      = v e(theta) dt + sqrt(2 D_T dt) xi_T,
Delta theta  =                  sqrt(2 D_R dt) xi_R,
P(tumble)    = 1 - exp(-lambda dt).
```

The TOML field widths and response coefficients are converted at the input
boundary. The stochastic recurrence, disk geometry, saved arrays,
diagnostics, and plots contain only dimensionless variables. No dimensional
trajectory or time array is returned.

## Linear response law

`kinetics.py` implements equation (2) literally:

```text
X(r, theta) = X0 + X_c delta_c + X_S S,
X in {D_T, D_R, v, lambda}.
```

For a canonical run, `Responses.only_case()` retains exactly one sensitivity.
Negative coefficients are errors. They are never silently clipped, because
clipping would replace the manuscript's linear model by an undocumented
nonlinear one.

## Continuous Itô update

For one dimensionless time step `dt`, coefficients are evaluated at the state
at the left endpoint:

```text
Delta r     = v e(theta) dt + sqrt(2 D_T dt) xi_T
Delta theta =                    sqrt(2 D_R dt) xi_R,
```

where `xi_T` has two independent standard-normal components and `xi_R` is
standard normal. This left-endpoint rule is essential: it is the Itô
interpretation that leads to `Delta(D_T P)` and `partial_theta^2(D_R P)` in
equation (11).

## Aggregate microscopic observables

The density ensemble also reduces saved particle states into moments before
each bounded batch is freed. No individual path is written to disk. The
reported quantities are

```text
mean concentration       <c(x(t))>
polar alignment          <e(theta) . grad(c)>
nematic alignment        <|grad(c)| cos(2(theta - phi_c))>
heading memory n=1       <cos(theta(t) - theta(0))>
heading memory n=2       <cos(2(theta(t) - theta(0)))>
cumulative tumbles       <N_tumble(t)>
displacement moments     <|x(t)-x(0)|^2>, <|x(t)-x(0)|^4>
```

For every observable `A`, the code forms the paired sample

```text
Delta A_m = A_m(case) - A_m(Case 0)
```

using the same random innovations for both paths. The saved uncertainty is
the ordinary standard error of the ensemble mean of `Delta A_m`, not the
standard error of two independent ensembles. This paired estimate is the
appropriate significance test for weak Case-minus-Case-0 responses.

The signed `Delta<c>` should be used to rank weak physical responses.
Pixelwise norms such as total variation are nonnegative even under a true
null response and therefore have a positive sampling-noise floor.

## Tumbles

With the pre-step rate frozen over `dt`, the probability of at least one
tumble is

```text
1 - exp(-lambda dt).
```

The code permits at most one tumble per step and enforces a configurable upper
bound on `lambda dt`. At the default maximum 0.05, two-or-more frozen-rate
events have probability below 0.00121; the actual defaults are much smaller.

The turn kernel is rotation invariant and achiral. It is constructed so that

```text
E[cos(turn angle)] = a1
```

exactly. Positive `a1` uses a wrapped normal around zero, negative `a1` uses a
wrapped normal around pi, and `a1 = 0` is uniform. Thus the first angular
moment is precisely the one in equations (7)-(8), while the code remains free
of a SciPy dependency.

## Timestep splitting

One step is:

1. evaluate `c`, `grad(c)`, `S`, and all kinetic coefficients at the old state;
2. apply the Itô positional and angular increments;
3. enforce the selected disk wall law;
4. test the pre-jump tumble event and, if it occurs, draw the new heading.

The relative order of continuous motion, wall contact, and a jump within one
finite step changes only terms beyond first order as `dt -> 0`.

## Canonical cases

| Case | Active sensitivity | Leading mobility contribution |
|---:|---|---:|
| 1 | `D_T_c` | `-D_T_c` |
| 2 | `D_R_c` | `0` |
| 3 | `v_c` | `-v0 v_c / (2 gamma0)` |
| 4 | `lambda_c` | `0` |
| 5 | `D_T_S` | `0` |
| 6 | `D_R_S` | `-v0 D_R_S / (2 gamma0)` |
| 7 | `v_S` | `v_S / 2` |
| 8 | `lambda_S` | `-v0 (1-a1) lambda_S / (2 gamma0)` |

The microscopic simulator includes all eight mechanisms even when their
leading macroscopic mobility is zero. The three zeros concern the particular
linear, slow-gradient closure, not the absence of microscopic effects.

`diagnostics.py` also evaluates

```text
gamma0 = D_R0 + lambda0 (1-a1)
D_eff  = D_T0 + v0^2 / (2 gamma0)
kappa  = v_S/2 - D_T_c
         - v0 [v_c + D_R_S + (1-a1) lambda_S] / (2 gamma0).
```

These are diagnostics only; the trajectory is generated from equation (6),
not from the macroscopic closure.

The microscopic dynamics allow `v0 = 0`, `D_R0 = 0`, `lambda0 = 0`, and zero
response coefficients. If `gamma0 = 0`, orientation is frozen unless another
configured mechanism changes it. The stochastic simulation remains
well-defined, but the long-time isotropic closure above does not apply, so
`D_eff` and `kappa` are displayed as `N/A`.

## Why a theory-zero case can be linear at a microscopic wall

The exact angularly integrated translational term is

```text
L_T rho = nabla^2[
    D_T0 rho + D_T_c delta_c rho + D_T_S p dot grad(c)
].
```

In the isotropic uniform bulk reference state, `p^(0)=0`. Cases 2 and 4 only
change the angular relaxation of `p`, and the Case 5 term above also vanishes
at first order. These are the cancellations behind their zero entries in the
closed bulk mobility.

The default sliding wall does not preserve an isotropic orientation
distribution at the boundary: outward-pointing swimmers accumulate until
angular diffusion or a tumble releases them. Thus `p_wall^(0)` is nonzero.
Case 2 or 4 can change that layer's relaxation at first order, while Case 5
contains the explicitly linear wall term
`D_T_S p_wall^(0) dot grad(c)`. The macroscopic no-flux condition is imposed
after this kinetic layer has been eliminated and is not an exact microscopic
collision law.

The included symmetric response-scaling audit confirms that the default
full-disk Case 5 signal is odd in its sensitivity and scales linearly. Its
retained *bulk* mobility remains zero; the two statements refer to different
asymptotic objects.

## Squared displacement versus MSD

For one realization, the displayed path has the squared displacement

```text
R_k^2(t) = |r_k(t) - r(0)|^2.
```

This is not yet an average. The statistically meaningful diffusion analogue
is computed from `M` independent replicates:

```text
MSD_k(t) = (1/M) sum_m |r_(k,m)(t) - r(0)|^2.
```

All replicates start at the disk center with the same heading. Within replicate
`m`, all eight cases use the same standard-normal increments and event-test
uniforms. This is a paired common-random-number comparison. It does not force
the physical increments to remain equal: for example,
`sqrt(2 D_T dt) xi_T` changes when a mechanism changes `D_T`, and the same
tumble-test uniform produces different events when `lambda` changes.

The dashed video reference is the exact free-space baseline result

```text
MSD_0(t) = 4 D_T0 t
           + 2 v0^2 [t/gamma0 - (1-exp(-gamma0 t))/gamma0^2].
```

It is ballistic at very short times and approaches `4 D_eff t` at long times.
The simulated curves need not follow it because the disk confines them and the
chemical field makes their kinetics position- and heading-dependent.

At `gamma0 = 0`, the continuous limiting form used by the code is

```text
MSD_0(t) = 4 D_T0 t + v0^2 t^2.
```

The phrase **free-space baseline** therefore does not mean Case 0. It is the
closed-form constant-coefficient MSD obtained after removing the disk wall.
Case 0 is simulated inside the disk and uses the configured wall law, but sets
all eight chemical-response coefficients to zero:

```text
D_T = D_T0,  D_R = D_R0,  v = v0,  lambda = lambda0.
```

## Evolving spatial probability density

The primary comparison now estimates

```text
p_k(r,t),  k = 0,...,8,
```

from `M` independent particles for each case. Within ensemble member `m`, all
nine cases use common random innovations. The reported density is

```text
p_k(x,t) = R^2 p_k,physical(r_physical,t_physical),
```

and integrates to one over the unit disk. Particle positions are accumulated
in a two-dimensional histogram, smoothed with a Gaussian kernel, corrected
for the missing kernel support outside the disk, and normalized at every
saved time. Case 0 is generated by `Responses.zeros()` and therefore uses
only the background (`_0`) kinetic coefficients.

The signed comparison uses

```text
delta p_k(x,t) = p_k(x,t) - p_0(x,t).
```

It is estimated from the same paired raw histograms with a separately declared
Gaussian bandwidth `b_delta >= b`. Thus smoothing occurs at the statistical
estimation stage, not as a filter on rendered pixels. Each constituent
`p_k` remains normalized, so the discrete integral of `delta p_k` is zero up
to floating-point roundoff. Each difference panel uses a standard linear map
from `[-L_k,+L_k]` to blue-white-red, with zero mapped to white. The case limit
`L_k` is fixed over all positions and times and selected from a configurable
robust percentile of `|delta p_k|`. This changes only rendered contrast; the
saved `delta p` values remain unchanged.
