# Numerical choices and accuracy

## Why each path is sequential

A single trajectory with state-dependent coefficients is causally sequential:
step `n+1` needs the position and heading from step `n`. NumPy pre-generates
all independent random arrays, while Numba compiles the complete recurrence to
native Arm64 code. Independent ensemble replicas are parallelized across
native threads; no attempt is made to parallelize time within one path.

## Choosing dt

Set `simulation.dt = "auto"` to scan the requested cases and enforce all of
the following limits:

```text
max(lambda) dt <= simulation.max_lambda_dt
sqrt(2 max(D_R) dt) <= numerics.max_rotation_rms
v_max dt <= f ell_min
sqrt(4 max(D_T) dt) <= f ell_min
```

Here `f = numerics.max_spatial_step_fraction`; the factor four is the
two-dimensional mean-square translational increment. `ell_min` is the minimum
of the disk radius, all built-in Gaussian widths, and an explicitly supplied
`numerics.characteristic_length`. The selected step is the largest permitted
by these parameter-derived constraints, reduced slightly so that an integer
number of steps reaches `duration` exactly.

A numeric manual `dt` is checked against the same limits and rejected when it
is too large. Automatic selection is a defensible starting point, not a
replacement for a convergence study: reduce the accuracy targets until the
observables of interest are stable.

## Checking the linear-theory regime

The terminal report evaluates three diagnostics tied to sections 2.4 and 5.2
of `Theory.pdf`:

```text
eta   = max |X-X0|/X0,
delta = (v0/gamma0)/ell_min,
gamma0 * t_final.
```

The asymptotic statements are `eta << 1`, `delta << 1`, and
`gamma0*t_final >> 1`; they do not define sharp cutoffs. The code reports, but
does not enforce, the conservative working targets `eta <= 0.15`,
`delta <= 0.2`, and `gamma0*t_final >= 10`.

`configs/default.toml` satisfies these working targets. Cases 2, 4, and 5
have zero *bulk* mobility at the order retained by the closure. Their
microscopic PDF differences may still contain a kinetic wall-layer response;
the response-scaling audit separates that model discrepancy from bulk
higher-order and Monte Carlo contributions.

## What is and is not discretized in the disk

The particle position is continuous; there is no spatial PDE mesh. A grid is
used to scan the disk before a run and another grid is used to draw the field.
Both are automatically required to have spacing

```text
Delta x <= ell_min / numerics.scan_points_per_length.
```

The scan evaluates the field on disk points and bounds
`S = e(theta) dot grad(c)` analytically over every possible heading. This
produces bounds on the velocity, both diffusivities, and tumble rate without
missing a dangerous heading between angular samples. If the required grid
would exceed `numerics.max_scan_grid`, the run stops and explains how to
increase the limit or smooth the field. For `kind = "user"`, the code cannot
infer unresolved structure, so `numerics.characteristic_length` is required.

## Disk wall

An attempted straight increment is intersected with the circle. The remaining
normal displacement is reflected, including repeated intersections if
needed.

- `sliding`: reflect position and leave heading unchanged. In the vanishing
  timestep limit this approximates a normally reflected positional process,
  while retaining the active heading that creates a kinetic wall layer.
- `specular`: reflect both position increment and heading about the local
  normal.
- `diffuse`: reflect position and draw a heading uniformly from the inward
  half-circle.

Finite-step reflection approximates the continuous reflected process. Near-wall
statistics should be checked especially carefully under time-step refinement.
The manuscript's warning still applies: as `D_T -> 0`, sliding walls develop
surface accumulation that may require an explicit surface-state model.

## Reproducibility and comparisons

For individual trajectories and MSD calculations, the seed creates fixed
arrays for translational noise, rotational noise, tumble tests, turn angles,
and diffuse-wall headings. All comparison cases reuse these innovations.
For the much larger PDF ensemble, the fused native kernel creates one
reproducible innovation set per particle and timestep and immediately applies
it to all nine cases. This avoids storing step-sized noise arrays while
preserving the same common-random-number comparison.

The readable density fallback now mirrors the fused kernel's scalar MT19937
draw order, including initialization. Engine comparisons therefore use the
same random streams. Native and readable histograms are bitwise identical in
the regression test; reduced floating-point moments agree to roundoff.

The probability-density calculation repeats this pairing independently across
an ensemble for Cases 0-8. Built-in fields use Numba native threads configured
by `performance.threads`. Positions are simulated in bounded-memory batches
and immediately accumulated into histograms; all `M` trajectories are never
retained simultaneously. Cases whose selected response coefficient is exactly
zero reuse Case 0's identical path instead of being simulated redundantly.

For KDE bandwidth `b R`, a uniform distribution contributes approximately
`2 M b^2` samples to one Gaussian-kernel footprint. The automatic ensemble
size is the next power of two that makes this value at least
`density.min_samples_per_kernel`. The defaults `b=0.05` and target `80`
therefore select `M=16384`. This is an occupancy/variance floor, not an
optimal-bandwidth rule or a proof of convergence. Increase `M` and vary both
bandwidths before making a quantitative claim.

The signed-difference KDE may use a larger
`density.difference_bandwidth_fraction`. Its samples-per-kernel diagnostic is
reported separately before simulation. In automatic mode the code uses no
more than twice the absolute bandwidth and preserves at least three kernel
widths across the narrowest resolved field length.

## Sampling and video

The integrator uses every `dt`; `save_every` only controls trajectory storage.
The plotted time coordinate is dimensionless. The renderer maps the complete
computed interval to at most ten wall-clock seconds, so changing `save_every`
does not change movie duration. For smooth sampling, choose

```text
save_every * dt <= 1 / fps.
```

The density movie has its own `density.video_fps`, `density.time_slices`, and
`density.max_video_seconds`, which cannot exceed 10. It displays the
dimensionless density `p(x,y,t)` directly with the same fixed `[0,1]` color
scale for all cases and times. It never divides the density by a frame-wise or
global maximum. Values above 1 saturate only in the rendered image; the saved
NPZ density remains unchanged.

A companion signed-difference movie displays
`delta p_k = p_k - p_0`. Each case uses a conventional linear signed scale
that is fixed over the complete movie. Its range is the configured
`density.difference_case_percentile` of `|delta p_k|` over the disk and all
times. Values outside that robust range saturate only in the image; the saved
numerical density is unchanged.
Its separately configured KDE bandwidth is applied directly to the same raw
histograms, not as a cosmetic frame filter. This makes perturbative changes
smoother without changing or renormalizing the absolute-density movie.

It uses direct Pillow rasterization into an FFmpeg raw-video pipe, avoiding a
full Matplotlib layout/render for every frame. Repeated density indices reuse
the already rasterized frame.
