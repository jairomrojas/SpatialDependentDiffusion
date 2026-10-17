# Performance and resolution guide

## Recommended profile

The production path is the Numba CPU engine with bounded batches. On the
20-vCPU WSL 2 host used for this audit, start with 8 native threads and a batch
of 256 particles, then measure nearby settings with the repository benchmark.
Do not assume that Linux CPU numbers 0--7 are physical P-cores: WSL exposes
virtual CPUs, and the host hypervisor retains the final placement decision.^1

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
  taskset -c 0-7 python -m analysis.simulations.performance_benchmark.run \
  --threads 8 --batch-size 64 256 1024 \
  --output outputs/simulations/population_release/benchmark-arrow-lake.json
```

Keep the project on the WSL ext4 virtual disk rather than `/mnt/c`; the current
workspace already satisfies that condition. On the reference workstation,
heavy jobs run under `science_run` inside a shared memory slice capped at
6 GiB (5 GiB soft limit, 1 GiB swap; updated 2026-10-13). Tests and individual
production stages fit; one full microscopic archive (~4.8 GB once decompressed)
nearly fills it, and loading two archives in one job will be killed.

## Measured optimization

The density estimator applies the same one-dimensional Gaussian kernel along
both spatial axes. The former implementation materialized a strided sliding
window and dispatched `tensordot`. A representative axis over a
`10 x 101 x 141 x 141` float64 stack measured 27.995 s. SciPy's compiled
`ndimage.convolve1d`, using the same weights, centered origin, and zero
extension, measured 1.367 s: a 20.5x kernel-level speedup with maximum absolute
difference `2.22e-16`. SciPy explicitly supports the output dtype and constant
boundary mode used here.^2

After replacement, the complete two-axis density smoothing and normalization
for that stack measured 0.684 s. The optimization has a dedicated convolution
equivalence test and a uniform-disk boundary-correction invariant. Numba output
reordering was also removed for the normal all-distinct response matrix, which
avoids copying positions, headings, and tumble counts after every batch.

An indicative one-run end-to-end sweep with 4,096 particles, 100 integration
steps, 101 saved slices, and a 141-square grid measured 13.322 s (batch 64),
11.766 s (batch 256), and 11.687 s (batch 1,024). Those numbers are not a
universal ranking: the production run is longer, thermal state matters, and
KDE finalization is batch-independent. Use at least three repeats for a tuning
decision.

## Memory model

For `C` cases, `T` saved times, grid width `G`, and float width `B`, one field
stack costs `C*T*G*G*B` bytes. At the canonical `C=10`, `T=301`, `G=141`, and
float64, each stack is 478,734,480 bytes (456.6 MiB). Grid refinement therefore
costs quadratically: doubling `G` requires four times the field memory and
roughly four times the spatial post-processing work.

The microscopic archive contains density, polarity, nematic, response, and
split-half error fields. Its current compressed file is about 3.4 GB and
expands to roughly 4.8 GB of array payload. The macro and analytic archives are
about 2.0 GB compressed each and roughly 2.9 GB raw each. These are valid local
caches, but they should not be committed. NumPy notes that compressed NPZ is a
ZIP of NPY members and recommends memory-mapped NPY or chunked formats for
arrays too large to fit comfortably in memory.^3

Practical controls:

- Reduce `density.time_slices` when only final or sparse reporting times are
  needed. This does not change the integration timestep.
- Keep `density.batch_size` small enough that positions, headings, and tumble
  counts remain cache-resident. At 256 particles, 10 cases, and 301 slices,
  those three arrays occupy about 20.7 MiB before temporary reducers.
- Increase ensemble size only when the paired standard error or explicit
  sampling-floor audit requires it. Four times as many independent particles
  halves Monte Carlo standard error.
- Increase `grid_size` only with a grid-convergence result. More pixels do not
  recover structure already removed by the observation bandwidth.
- Keep raw archives on the Linux filesystem. Publish validated archives as a
  versioned release or data deposit, not inside Git history.^4

## CUDA assessment

The RTX 5070 is not currently used by `chem_swimmer`; installing PyTorch or a
CUDA runtime does not accelerate a CPU-only Numba kernel. The modern
`numba-cuda` package supports CUDA 12 devices through compute capability 12.1,
so sm_120 is supported by the toolchain.^5 However, Numba-CUDA is now in
maintenance mode and its maintainers direct new development toward
Numba-CUDA-MLIR.^6

A worthwhile GPU backend must keep the full loop on device: generate random
innovations, advance all paired cases, and reduce directly into histograms and
observables. Copying every saved particle state over PCIe would discard much of
the benefit; NVIDIA's guidance prioritizes minimizing host/device transfers,
coalescing global accesses, and avoiding long divergent branches.^7 The
sliding-wall and rejection/tumble branches make this a validation project, not
a decorator change. Required acceptance gates are fixed-seed distributional
parity, mass conservation, boundary invariants, response-sign tests, and an
end-to-end speedup including transfers. Until that work is funded and measured,
the CPU path remains the reliable production backend.

## Sources

1. Microsoft WSL, “[Hybrid CPU discussion #9190](https://github.com/microsoft/WSL/discussions/9190),” 2022--2024; Microsoft WSL, “[Hybrid topology issue #13272](https://github.com/microsoft/WSL/issues/13272),” 2025.
2. SciPy, “[`scipy.ndimage.convolve1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html),” SciPy 1.18 documentation.
3. NumPy, “[Reading and writing files](https://numpy.org/doc/stable/user/how-to-io.html),” NumPy 2.5 documentation.
4. GitHub, “[Repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits),” accessed September 2026.
5. NVIDIA, “[Numba-CUDA installation](https://nvidia.github.io/numba-cuda/user/installation.html),” accessed September 2026.
6. NVIDIA, “[Numba-CUDA maintenance notice](https://nvidia.github.io/numba-cuda/),” accessed September 2026.
7. NVIDIA, “[Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html),” CUDA 12.8; NVIDIA, “[CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),” accessed September 2026.
