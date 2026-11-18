# LNPCC — Particle Competition and Cooperation for Semi-Supervised Learning with Label Noise

MATLAB and C/MEX implementation of **Label-Noise Particle Competition and Cooperation (LNPCC)**, a graph-based semi-supervised learning approach designed to work with noisy input labels.

This repository accompanies the article:

> Breve, Fabricio Aparecido; Zhao, Liang; and Quiles, Marcos Gonçalves. "Particle Competition and Cooperation for Semi-Supervised Learning with Label Noise." *Neurocomputing*, **160**, 63–72 (2015). https://doi.org/10.1016/j.neucom.2014.08.082

- [Published article (DOI)](https://doi.org/10.1016/j.neucom.2014.08.082)
- [arXiv preprint (posted in 2020)](https://arxiv.org/abs/2002.05198)
- [Author-hosted accepted manuscript (PDF)](https://www.fabriciobreve.com/artigos/2015/Neurocomputing2015.pdf)

## Overview

LNPCC extends the [Particle Competition and Cooperation (PCC)](https://github.com/fbreve/Particle-Competition-and-Cooperation) framework to classification with potentially incorrect initial labels. Unlike the original PCC implementation, the propagation routine can update dominance values at initially labeled vertices, allowing the network to mitigate label noise.

The current implementation constructs a graph that prioritizes connections between vertices with the same provided label, then performs **ten particle-propagation executions**. Its output class assignments are determined from the accumulated dominance values of those executions.

**Historical note:** The algorithm was published in 2015. The code in the current `master` branch has changed since the initial release, particularly in December 2025 and January 2026. Do not assume this branch is an exact copy of the code or numerical behavior used for the published experiments.

## Requirements

- **MATLAB R2021a or newer** for the `name=value` function-call syntax in `example.m` (the `arguments` block in `lnpcc.m` was introduced earlier).
- **Statistics and Machine Learning Toolbox**, used for nearest-neighbor search and other utilities (such as `knnsearch`, `pdist`, `squareform`, and `random`).
- The `lnpccloop` **MEX** routine, which is mandatory in the current implementation. There is no `mex=false` pure-MATLAB execution path.

A precompiled `lnpccloop.mexw64` is included for **64-bit Windows**. Other operating systems, or incompatible MATLAB installations, need to compile `lnpccloop.c` with a supported C compiler:

```matlab
mex -setup C
mex lnpccloop.c
```

Compiling and running have not been validated here for every MATLAB version or operating system.

## Quick start: Wine with label noise

The repository includes `wine.data` and an example using the Wine dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine).

Open MATLAB, set the working directory to the repository root, ensure the MEX routine is available, and run:

```matlab
example
```

The example generates a partially labeled dataset with **10% labeled samples, of which 30% receive incorrect labels** (subject to rounding and the label-generator's constraints). It runs LNPCC with `k=10` and standardized Euclidean distance, and reports classification accuracy for the originally unlabeled examples and execution time.

A direct call has the form:

```matlab
[owner, pot, owndeg, distnode] = lnpcc(X, uint16(slabel), ...
    k=10, disttype="seuclidean");
```

Here, each row of `X` is a sample; the `slabel` vector uses `0` for unlabeled samples and consecutive positive class indices for labeled samples.

## Main options

| Option | Meaning |
| --- | --- |
| `k` | Nearest-neighbor count; default approximately 5% of the dataset size |
| `disttype` | Distance metric used for graph construction (default `"euclidean"`) |
| `earlystop` | Whether to use an early-stopping criterion (default `true`) |
| `valpha` | Controls the early-stopping stability window (default `2000`) |
| `maxiter` | Maximum iterations per execution (default `500000`) |
| `pgrd`, `deltav`, `deltap`, `dexp` | Particle movement and dominance update parameters |
| `nclass` | Number of classes (default: maximum provided label) |
| `useseed`, `seed` | Control the C random-number generator used by the MEX routine |
| `legacy_knn` | Use the original full-distance-matrix graph construction (default `false`) |
| `scale_scope` | Optimized `seuclidean` construction: `"global"` (corrected default) or `"subset"` (2025 compatibility) |
| `reseed_each_run` | Restore legacy seeded behavior when `true`; default is to seed only the first run |

### Reusing a prebuilt graph

A graph can be built once and reused for repeated propagations with **the same feature data and the same noisy-label vector**:

```matlab
G = lnpcc_buildgraph(X, uint16(slabel), k=10, ...
    disttype="seuclidean", scale_scope="global");
owner = lnpcc(G, uint16(slabel), Xgraph=true, ...
    useseed=true, seed=1234);
```

The graph retains the label vector used to construct it. LNPCC checks this against the supplied `slabel` and **rejects mismatches**, so a graph must be rebuilt when the labeled set or its noisy labels change. In this mode, `k` comes from the graph and the `legacy_knn` and `scale_scope` arguments used to build it must be set in `lnpcc_buildgraph`.

The current code requires the compiled MEX; there is no `mex=false` mode.

### Outputs and reproducibility caveats

- `owner`: predicted classes, obtained from the **sum of the dominance matrices across ten executions**.
- `pot`: the dominance matrix from the **last execution**, not the sum used to compute `owner`.
- `owndeg`: normalized fuzzy ownership accumulated across the ten executions (following the January 2026 update).
- `distnode`: particle distance matrix from the last execution.

The C/MEX routine uses `rand()` and `srand(seed)`. **With `useseed=true`, the current implementation seeds the C generator once before the first of ten executions and lets the same stream advance during subsequent executions.** Repeated calls with the same seed should therefore reproduce the full experiment without repeating the identical random stream ten times. For compatibility with the pre-correction behavior, set `reseed_each_run=true`, which resets the same seed at the beginning of each execution.

The newer graph builder uses `knnsearch`, while `legacy_knn=true` uses the older full-distance method. For `disttype="seuclidean"`, the December 2025 optimized builder computed standard deviations **separately for label-based subsets**, unlike the legacy builder's global scaling. This changed neighbors even without distance ties. The corrected **default** `scale_scope="global"` standardizes features once using the full dataset; choose `scale_scope="subset"` to reproduce the 2025 optimized behavior. Agreement of graph structure in one dataset does not prove equivalence to the published 2015 experiments.

## Files

| File | Purpose |
| --- | --- |
| `lnpcc.m` | Main algorithm, graph selection, and aggregation of the ten executions |
| `lnpcc_buildgraph.m` | Reusable label-aware graph construction (optimized/legacy, with scaling options) |
| `compare_lnpcc_graphs.m` | Diagnostic comparison of graph structures using Wine |
| `tests/test_lnpcc_contracts.m` | Regression tests for graph contracts, seed behavior, and prebuilt graph parity |
| `lnpccloop.c` | C source for particle propagation |
| `lnpccloop.mexw64` | Precompiled MEX routine for 64-bit Windows |
| `example.m` | Wine dataset demonstration with noisy labels |
| `slabelgenwl.m` | Partially labeled data generator with injected label noise |
| `stmwevalk.m`, `kappa.m`, `sumcol.m` | Evaluation and supporting utilities |
| `wine.data`, `wine.names` | Third-party Wine example dataset and description |
| `changelog.txt` | Notes on post-publication changes |

The `.asv` files are historical MATLAB/editor backups retained for reference; they are not required to run the example.

### Validation

From the repository root, after compiling `lnpccloop.c` for the current platform, run:

```matlab
results = runtests("tests");
disp(results);
report = compare_lnpcc_graphs();
```

Tests involving the MEX routine are skipped if it is unavailable. The graph diagnostic reports differences in **edges**, not accuracy, and does not assume the constructions are equivalent.

**Validation status:** These tests and MATLAB/MEX scientific executions must be run in a compatible MATLAB environment. This repository change does not claim a successful experimental validation yet.

## Historical source snapshot

- [`archive/2019-initial-release`](https://github.com/fbreve/LNPCC/tree/archive/2019-initial-release) points to commit [`48c1a8ac4ccc`](https://github.com/fbreve/LNPCC/commit/48c1a8ac4ccc4cebcb3fc8da368a770ccdf06352), the earliest complete source release in Git, dated **January 24, 2019**.
- The journal paper is from **2015**. No matching 2015 source-code snapshot has been established from this repository. The 2019 branch must **not** be described as a proven copy of the experimental implementation used for the publication.
- The archived branch is left unchanged. It contains its original MATLAB files, MEX binary, and Wine data. Current bug fixes and regression tests belong on `master` after scientific validation, not on the archival branch.

To reproduce a specific subsequent result, use the corresponding commit in Git history together with its exact MEX build, MATLAB environment, noise labels, random seeds, and other experiment settings.

## Citation

If you use LNPCC in research, please cite the original journal article:

```bibtex
@article{breve2015particle,
  author  = {Breve, Fabricio Aparecido and Zhao, Liang and Quiles, Marcos Gon{\c{c}}alves},
  title   = {Particle Competition and Cooperation for Semi-Supervised Learning with Label Noise},
  journal = {Neurocomputing},
  volume  = {160},
  pages   = {63--72},
  year    = {2015},
  doi     = {10.1016/j.neucom.2014.08.082}
}
```

## License and data

The project source code is distributed under the [Apache License, Version 2.0](LICENSE). This does **not** automatically extend the software license to the third-party Wine dataset or any other externally sourced material. Their respective usage and attribution conditions remain applicable.
