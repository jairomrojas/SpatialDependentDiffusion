# Repository engineering audit

Paths below are as of the audit; since 2026-10-16 the package lives in `model/solvers/chem_swimmer/`, scripts in `analysis/`, results in `outputs/` (see the root `README.md`), and it is no longer packaged for installation.

## Executive assessment

The numerical core is unusually rigorous for a research repository. It has a
readable reference integrator, paired random innovations, conservative
finite-volume solvers, explicit stability plans, provenance-bearing archives,
and tests for mathematical identities rather than only software behavior. The
main risks were execution cost, environment drift, generated-data scale, and a
repository boundary that did not match the scientific project boundary.

The implemented changes address the highest-confidence items: compiled bounded
memory convolution, elimination of redundant batch copies, atomic archive and
manifest replacement, non-pickle archive loading, documented global Pixi usage,
format enforcement, a four-version CI test matrix, package build verification,
machine-readable citation metadata, and a reproducible hardware benchmark.
The baseline rendering defect is fixed: two-method comparisons now select the
two-method renderer instead of entering the three-method layout and failing.

The remaining high-priority decisions require owner authority or a distinct
engineering project: select a software license, publish large validated data
under persistent identifiers, and implement a reduced-on-device CUDA backend
only if an end-to-end prototype beats the tuned CPU route.

## Audit matrix

| Area | Finding | Action | Status |
|---|---|---|---|
| Correctness | Two-method manuscript rendering entered a three-method-only layout. | Dispatch to the existing two-method layout and retain the failing integration test as regression coverage. | Complete |
| Performance | Sliding-window/tensor contraction made KDE convolution the dominant post-processing cost. | Replace with compiled `scipy.ndimage.convolve1d`; verify to machine precision. | Complete |
| Performance | Distinct response matrices were sorted/deduplicated and then copied back after every native batch. | Return native outputs directly when every response row is unique. | Complete |
| Durability | A killed multi-gigabyte save could replace a useful cache with a truncated NPZ or manifest. | Write in the destination directory, flush/fsync, and atomically replace. | Complete |
| Security | NPZ readers accepted NumPy's pickle-enabled default despite storing only primitive arrays. | Load with `allow_pickle=False` and close every archive deterministically. | Complete |
| Reproducibility | The workstation intentionally uses the global Pixi `physics` environment, while package bounds allowed incompatible releases. | Document the mandatory environment and tighten interoperable package bounds; retain standard `venv` instructions for external collaborators. | Complete |
| Compatibility | `numpy>=2.0` could resolve alongside SciPy 1.11, which predates NumPy 2 support; the Numba cap excluded the installed/tested 0.67 release. | Require SciPy 1.13+ and Numba 0.67.x; cap NumPy below 2.6 per Numba's compatibility table. | Complete |
| Quality | Ruff lint passed after one fix, but 34 files failed the formatter. | Format the tree and enforce lint plus format in CI. | Complete |
| CI | Only Python 3.11 and 3.13 were tested; installed CLI and distributable build were not checked. | Test 3.11--3.14, smoke-test the CLI, and build wheel/sdist. | Complete |
| Distribution | An installed wheel omitted the default TOML files and resolved output paths inside the environment. | Ship the three configurations as data files; use the caller's directory for installed outputs. | Complete |
| Discoverability | Package metadata lacked authors, keywords, URLs, and citation metadata. | Add PEP 621 metadata and `CITATION.cff`. | Complete |
| Data scale | Local reproducible outputs occupy 7.1 GB; one Git commit would make the repository unhealthy. | Keep raw data ignored and document release/deposit handling. | Complete locally; deposit pending |
| Licensing | No license grants reuse rights. | Owner must select and add a license; do not infer legal intent. | Blocked on owner choice |
| Repository boundary | The outer scientific tree contained the manuscript and simulation, but only `simulation/` had Git history. | Move the existing metadata to the outer root, prefix the former project tree under `simulation/`, retain the remote and history, and restore staged/unstaged state. | Complete |
| GPU | CUDA is installed, but no source imports or dispatches a CUDA backend. | Treat CUDA as a future backend with explicit validation and benchmark gates. | Scoped roadmap |

## Correctness and validation

The baseline suite completed 135 tests and 79 subtests before failing in the
last comparison-rendering integration. The cause was semantic, not graphical:
`render_manuscript_strategy_figures` always called the three-column Case 1
renderer, while its public contract and fixture permit two methods. Dispatch is
now based on method count, and the formerly failing test passes.

The density optimization preserves the exact discrete kernel definition. A
new test compares the compiled convolution with a zero-padded NumPy reference,
including uint32 histogram input and explicit float64 output. A second test
asserts that support-corrected smoothing maps a uniform disk histogram to a
uniform interior density. Existing engine-parity tests continue to compare the
readable and Numba particle paths, including density, polarity, nematic fields,
initial conditions, and paired errors.

Final verification used `/home/inre/.pixi/envs/physics/bin/python` under the
eight-vCPU affinity mask. It completed 138 tests and 79 subtests in 246.21 s
with no failures. Ruff lint and format checks, byte compilation, TOML parsing,
CLI startup, and wheel/source-archive builds also passed. The distribution
check confirmed that the installed wheel carries the default configuration and
that the source archive carries configurations, documentation, scripts, and
tests. Repository citation metadata now lives at the expanded Git root, where
GitHub can discover it.

Scientific changes should continue to require a limiting case, conservation or
positivity checks, fixed-seed reproducibility where meaningful, and refinement
in the relevant control. This matches the repository's own contribution
checklist and the FAIR4RS emphasis on reusable software carrying clear identity,
access, interoperability, and reuse metadata.^1

## Performance findings

The major speed opportunity was not CUDA or more threads. Profiling the
representative KDE stack isolated a 20.5x axis-level speedup by expressing the
same separable operation through SciPy's compiled one-dimensional convolution.
This follows Numba's general guidance to keep loops in compiled form and use
explicit parallelism only where it maps cleanly to independent work.^2 It also
avoids exposing a very large virtual sliding-window tensor to a contraction
backend.

Thread counts should be empirical. Intel recommends P-core-only execution as a
simple predictable policy for heterogeneous CPUs,^3 but WSL virtualizes CPU
topology and its issue tracker documents that guest affinity does not prove a
particular host core type.^4 Therefore `taskset -c 0-7` is a useful eight-vCPU
cap, not evidence by itself that eight Lion Cove cores executed the work. The
new benchmark records affinity, CPU, Python, NumPy, SciPy, Numba, workload, and
all repeat timings so the choice is auditable.

The `memory=11GB` WSL limit is reasonable for keeping Windows responsive and
the measured tests peaked at about 2.1 GB. Microsoft documents that `memory`
caps the VM, `swap` provides disk-backed overflow, `autoMemoryReclaim=gradual`
slowly returns cached memory, and `sparseVhd=true` affects newly created VHDs.^5
The main memory risk is simultaneous decompression of all production archives,
whose raw array payload approaches the VM cap. Longer term, split NPY or a
chunked scientific format should replace monolithic compressed NPZ for
interactive multi-method analysis; NumPy specifically points large-array users
to memory mapping and formats such as Zarr, HDF5, or NetCDF.^6

Higher resolution should follow error evidence, not available hardware alone.
For Monte Carlo estimates, quadrupling independent particle count halves the
standard error. For raster fields, doubling grid width quadruples memory and
spatial work. Increasing saved slices raises archive memory linearly but does
not improve timestep accuracy. The correct sequence is: timestep refinement,
ensemble/bandwidth refinement, grid refinement, and only then a production
profile that meets the observed error budget.

## Environment and packaging

The former dependency interval allowed combinations that were not mutually
supported. SciPy 1.13 was the release explicitly made for NumPy 2.0 support,^7
while Numba 0.67 supports Python 3.10--3.14 and NumPy 2.0--2.5.^8 The package
metadata now expresses those intersections. The mandatory global `physics`
environment currently supplies Python 3.12.14, NumPy 2.5.3, SciPy 1.18.0, and
Numba 0.67.0. Because global Pixi manifests live outside the repository, the
generation environment remains captured in artifact provenance and external
collaborators install from the declared package requirements.

CI separates quality from runtime testing. Ruff lint and formatting run once;
the scientific suite runs on Python 3.11 through 3.14; an installed CLI
smoke test checks entry-point packaging; and a dedicated job builds both wheel
and source distribution. GitHub's own Python Actions guidance recommends a
version matrix and dependency caching for this class of project.^9

## Data, provenance, and publication hygiene

Raw `results/data/` archives are already ignored, which is correct. GitHub
recommends storing programmatically generated files outside Git and using LFS
for binary content that truly belongs in repository history; it enforces a
100 MB object ceiling and recommends much smaller objects for repository
health.^10 The present 1.9--3.6 GB archives belong in a release asset or
research-data repository, accompanied by checksums, configuration, schema,
software version, and generation command.

The existing archive provenance is strong: configuration text and SHA-256,
schema, runtime versions, method, seed, timestep, and estimator settings are
embedded. Atomic replacement now prevents interrupted jobs from corrupting the
last valid cache. The next deposit should add a top-level inventory with file
SHA-256 values and persistent DOI. `CITATION.cff` is now present so GitHub can
display citation guidance and Zenodo can ingest software metadata.^11

No license has been added. Citation metadata credits authorship, but it does
not grant permission to copy, modify, or redistribute. The owners should select
a license compatible with the manuscript, dependencies, institutional policy,
and intended reuse before calling the repository public-release ready.

## CUDA roadmap

The current application is CPU-only despite a working CUDA/PyTorch/Numba
toolchain. `numba-cuda` supports CUDA 12 through compute capability 12.1,^12 but
the project has no CUDA kernels and should not advertise GPU acceleration.
Moreover, the Numba-CUDA project is in maintenance mode and recommends
Numba-CUDA-MLIR for forward development.^13

The execution sandbox used for this audit denied NVML/device access, so no GPU
timing is presented as measured evidence. The CUDA section is a code-path audit
and implementation roadmap; it does not contradict the workstation's separate
toolchain verification.

The Monte Carlo loop is parallel enough to justify a prototype, but it should
be designed around reduction rather than trajectory export. Each device thread
can own one particle, share innovations across its ten response cases, and
accumulate block-private histograms/observables before a global reduction. The
state layout must make adjacent threads access adjacent data, transfers must be
limited to reduced outputs, and branch divergence at rejection sampling,
sliding-wall contact, and tumbling must be measured. Those requirements follow
NVIDIA's Blackwell priorities: coalesced access, minimal redundant global
memory traffic, minimal transfers, and avoidance of long divergent paths.^14

Acceptance should be based on wall-clock time for the entire
simulate--reduce--smooth pipeline, not kernel time. Statistical parity should
be evaluated with distributions and observables rather than bitwise RNG parity,
while deterministic PDE paths remain float64. If a GPU prototype cannot exceed
the tuned CPU pipeline after transfers and validation, it should remain an
experimental branch.

## Release checklist

1. Verify the expanded root and all relative manuscript links in a fresh clone
   after pushing the root-expansion commit.
2. Choose a license and add its SPDX identifier to `simulation/pyproject.toml` and
   `CITATION.cff`.
3. Activate the global `physics` environment and run `make verify` in a clean
   checkout; record `pixi global list` with the release artifacts.
4. Run the numerical convergence, response scaling, sampling-floor, and
   long-time validation scripts for any scientific change.
5. Regenerate figures and metrics from one configuration hash; compile the
   manuscript from a clean tree.
6. Publish raw archives externally with checksums and a DOI; update the result
   inventory and citation metadata.
7. Tag a version matching `_version.py`, `CITATION.cff`, and archive provenance.

## Sources

1. FAIR for Research Software Working Group, “[FAIR Principles for Research Software (FAIR4RS Principles) 1.0](https://www.pure.ed.ac.uk/ws/portalfiles/portal/280109758/FAIR4RS_Principles_Final_Recommendation_Zenodo.pdf),” 2022.
2. Numba, “[Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html),” accessed September 2026.
3. Intel, “[Managing Performance with Heterogeneous Cores](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/managing-performance-with-heterogeneous-cores.html),” oneMKL Developer Guide.
4. Microsoft WSL, “[Hybrid CPU discussion #9190](https://github.com/microsoft/WSL/discussions/9190)” and “[Hybrid topology issue #13272](https://github.com/microsoft/WSL/issues/13272).”
5. Microsoft, “[Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config),” updated 2026.
6. NumPy, “[Reading and writing files](https://numpy.org/doc/stable/user/how-to-io.html),” NumPy 2.5 documentation.
7. SciPy, “[SciPy 1.13.0 Release Notes](https://docs.scipy.org/doc/scipy-1.13.0/release/1.13.0-notes.html),” 2024.
8. Numba, “[Version support information](https://numba.readthedocs.io/en/latest/user/installing.html),” accessed September 2026.
9. GitHub, “[Building and testing Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python),” accessed September 2026.
10. GitHub, “[Repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits),” accessed September 2026.
11. Zenodo, “[CITATION.cff file](https://help.zenodo.org/docs/github/describe-software/citation-file/),” accessed September 2026.
12. NVIDIA, “[Numba-CUDA installation](https://nvidia.github.io/numba-cuda/user/installation.html),” accessed September 2026.
13. NVIDIA, “[Numba-CUDA](https://nvidia.github.io/numba-cuda/),” maintenance notice, accessed September 2026.
14. NVIDIA, “[Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html),” CUDA 12.8.
