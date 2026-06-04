# UMAP (Go) — Engineering Deep-Dive & Handoff

**Repo:** fork of `github.com/nozzle/umap` (BSD-3-Clause)
**Branch with our work:** `claude/festive-brahmagupta-UQGr9` (HEAD `176904a`; hardening in `0698a1d`)
**Module path:** `github.com/nozzle/umap` · **Go:** 1.25 · **External deps:** 1 (`gonum`)
**Size:** ~5,400 LOC Go across 11 packages.
**Audience:** the engineer lifting this into another repo and finishing it for v1.

This document is long on purpose. §1–2 are orientation, §3–6 are the technical findings with
file references, §7 is the lift mechanics, §8 is the appendix (config, metrics, commands).

---

## 1. What this project is

A **pure-Go implementation of UMAP** (Uniform Manifold Approximation and Projection), a non-linear
dimensionality-reduction algorithm. You give it a high-dimensional numeric matrix (one row per item,
~15–25 columns in our case) and it returns a low-dimensional embedding (2 or 3 coordinates per row)
where similar rows land near each other.

**How UMAP works, in four stages** (this maps 1:1 onto the package layout, which makes the code easy
to follow):

1. **k-NN graph** — for every row, find its *k* nearest neighbors in the original space (`nn/`).
2. **Fuzzy simplicial set** — turn those neighborhoods into a weighted graph whose edge weights are
   "probabilities that two points are connected," with per-point bandwidth normalization (`graph/`).
3. **Initialization** — choose a starting low-dimensional layout, either from the graph's spectrum or
   randomly (`init/`).
4. **Layout optimization** — run stochastic gradient descent that pulls connected points together and
   pushes unconnected points apart, until the 2D/3D layout settles (`layout/`).

**Our use of it:** the offline reduction step in a carrier/broker "similarity cloud." We assemble a
feature matrix keyed by USDOT, UMAP reduces it to 3 coordinates, and a WebGL point cloud renders it.
It runs **offline / batch, never on a user request.** That property is the whole reason an unproven
all-Go library is an acceptable bet here: a bad layout is visible immediately, and the blast radius of
a defect is one batch job, not production traffic.

**Top-level API** (`umap.go`):
```go
cfg := umap.DefaultConfig()      // see §8 for every field
cfg.NComponents = 3
cfg.Metric = "euclidean"
cfg.Seed = 42                    // reproducibility
m := umap.New(cfg)
embedding := m.FitTransform(data) // [][]float32, shape n × NComponents, in input-row order
```
Row order is preserved, so `embedding[i]` corresponds to `data[i]` — you re-key back to USDOT by index.

---

## 2. Architecture & data flow (package by package)

The dependency graph is a clean DAG with one external edge:

```
umap (root: umap.go)
   ├─ nn/        ─→ distance/, internal/{heap, parallel, rand}
   ├─ graph/     ─→ internal/parallel
   ├─ init/      ─→ graph/, internal/rand,  gonum/v1/gonum/mat   ← the ONLY external dep
   ├─ layout/    ─→ graph/, internal/{parallel, rand}
   └─ distance/  ─→ (stdlib only)
internal/{heap, math, parallel, rand} ─→ leaves (stdlib only)
cmd/umap/        ─→ root (CLI; not needed as a library)
```

### The call chain through `Fit()` (`umap.go`, `func (u *UMAP) Fit`)
This is the spine of the whole library; reading these five calls in order explains everything:

1. `u.buildKNNGraph(data)` → **`nn.BruteForceKNN`** for `n < 1000`, else **`nn.NNDescent`**
   (`umap.go`, `buildKNNGraph`). Returns `*nn.KNNGraph` (`Indices [][]int32`, `Distances [][]float32`).
   ⚠️ The `NNDescent` branch is where the race lives (§3).
2. `u.buildFuzzySimplicialSet()` → **`graph.FuzzySimplicialSet`** (`graph/fuzzy.go:62`). Produces a
   `*graph.CSRMatrix` (sparse). The per-point bandwidth search is `smoothKNNDist`
   (`graph/fuzzy.go:142`) — a 64-iteration binary search that **faithfully mirrors** Python's
   `smooth_knn_dist`, including the `knn_dists[:, 1:]` self-skip and `local_connectivity` interpolation.
3. Embedding init via MT19937 (`u.initializeEmbeddingWithRNG`) → **`init.InitializeEmbedding`**
   (`init/spectral.go:154`). Spectral path builds a dense Laplacian and calls `gonum` `mat.EigenSym`;
   **random path** uses the NumPy-compatible MT19937.
4. RNG-state derivation for SGD (`umap.go`, in `Fit`): pulls three `int32`s from the same MT19937 to
   match Python's `random_state.randint(...)` — this is careful reference-fidelity work.
5. `u.optimizeLayoutWithRNGState(...)` → **`layout.OptimizeLayout`** (`layout/optimize.go:276`). The
   SGD loop. Curve params from `findABParams` (`layout/optimize.go:68`); per-edge work in
   `processEdgePython` (`layout/optimize.go:~380`).

### Package notes (with the load-bearing details)

**`distance/`** — a broad metric registry (`distance/distance.go:28`, `Registry`): euclidean, manhattan,
cosine, correlation, chebyshev, canberra, braycurtis, haversine, hellinger, plus a full binary set
(hamming, jaccard, dice, …). All wrap stdlib `math` (`sqrt32`/`pow32` at `distance.go:103-109`).
**Important subtlety:** the input `Metric` only affects **stage 1** (the k-NN graph). The SGD layout
(stage 4) always optimizes *Euclidean* distance in the embedding space — `processEdgePython` hardcodes
the euclidean output gradient. That is algorithmically correct (this is what Python UMAP does too), but
it means the entire **`GradRegistry`/`GetGrad`/`GradFunc` machinery is unused dead code**
(`distance.go:64-95`; see §5).

**`graph/`** — `CSRMatrix` (`fuzzy.go:22`) is the sparse graph type used everywhere downstream.
`FuzzySimplicialSet` parallelizes the sigma/rho computation safely (each worker writes a disjoint index,
`fuzzy.go:83`). ⚠️ **Scale watch:** `fuzzySetUnion` (`fuzzy.go:297`) symmetrizes by building Go
**maps** keyed on `(i,j)` over all `n·k` edges — fine at v1, but map allocation/GC will be the second
scaling bottleneck after NN (it materializes the full edge set in a hash map). `ToEpochsPerSample`
(`fuzzy.go:366`) and `GetEdges` (`fuzzy.go:398`) feed the SGD loop. Uses real stdlib `math.Exp`
(`fuzzy.go:112,211`) — **no hand-rolled math here.**

**`init/`** — `SpectralEmbedding` (`spectral.go:28`) builds a **dense** `n×n` normalized Laplacian and
runs `gonum` `mat.EigenSym` — O(n²) memory, O(n³) time. **It silently falls back to random init for
`n > 5000`** (`spectral.go:57-60`) and on any eigensolver failure (`spectral.go:94`). Net effect for us:
**at v1 scale you are on random init, not spectral.** `scaleAndCenter` (`spectral.go:171`) rescales to
[-10,10], which differs from Python's spectral noise scaling (another reason exact Python match is not a
goal). This file is the **only** consumer of `gonum`.

**`layout/`** — the core. `OptimizeLayout` (`optimize.go:276`): computes `a`/`b` curve params
(`findABParams:68`, a Levenberg–Marquardt fit), builds per-edge sampling schedules from the graph, and
runs `NEpochs` of SGD. Per-vertex RNG states are seeded à la Python
(`rng_state_per_sample`, `optimize.go:~321`, including a bit-level `unsafe.Pointer` float64→int64 cast
`float64ToInt64Bits`). Gradient signs, **±4 clipping** (`clip`, `optimize.go:~536`), learning-rate decay
`α(1−e/E)`, and the negative-sampling schedule all **match the reference** — verified, not assumed.
⚠️ **Hand-rolled math:** this file defines its own `fastExp/fastPow/fastLog` (`optimize.go:226-273`) and
`exp32/log32/pow32/sqrt32` (`optimize.go:~547-616`) using fixed-length Taylor / arctanh series. These
are **less accurate** (the `log32` arctanh series converges slowly for large arguments — exactly the
regime of large embedding distances) **and slower** than stdlib. See §4.

**`internal/parallel/`** — `ParallelFor(start,end,n,fn)` (`parallel.go:15`) chunks `[start,end)` across
`n=GOMAXPROCS` goroutines, each owning a contiguous range. Race-free **only if `fn(i)` writes solely to
index `i`'s data.** That invariant holds in `graph` but is **violated in `nn` (§3).**

**`internal/rand/`** — two RNGs: `tau.go` (Tausworthe `tau_rand_int`, `tau.go:28`, matches Python's
SGD RNG) and `mt19937.go` (NumPy `RandomState`, used for init). ⚠️ Two issues here: (a) `Intn`
(`tau.go:57`) is biased — `int(i) % n` with no rejection sampling; (b) `NormFloat64` (`tau.go:79`)
carries **another** copy of hand-rolled `sqrt/log/cos` (`tau.go:93-138`) and is **completely unused**
(§5).

**`internal/heap/`** — fixed-size max-heaps for k-NN (`SimpleHeapPush`, `FlaggedHeapPush`,
`DeheapSort`). `FlaggedHeapPush` (`maxheap.go:~245`) is where the NN-descent race manifests
(`maxheap.go:253` read, `:263` write).

**`internal/math/`** — stdlib-backed float32 wrappers (`math32.go`: `Exp32`, `Log32`, `Pow32`, …).
**Currently imported by nobody** (0 references). This is the good code that the hand-rolled math in
`layout/` and `rand/` *should* be calling. It's the drop-in target for the §6 step-2 fix.

---

## 3. Reproducibility & concurrency analysis (the one v1 blocker)

**Requirement:** same input + same seed → same layout (users revisit the cloud; reshuffling erodes trust).

**Status:** ✅ holds for `n < 1000`; ❌ **fails for `n ≥ 1000`** — and v1 is tens of thousands of rows.

**Empirical proof we ran:**
- `n = 600` (brute-force path), same seed, two runs → **byte-identical**.
- `n = 1200` (NN-descent path), same seed, two runs → **different output**.
- `go test -race ./nn/...` → **fires `DATA RACE`** (reproducible, not theoretical).

**Root cause** (`nn/nndescent.go`, `nnDescentUpdate`):
- `parallel.ParallelFor` runs `func1` across goroutines (`nndescent.go:237`), but inside it
  `tryUpdateNeighbors(... p1, p2 ...)` calls `FlaggedHeapPush` that **writes `indices[j]/distances[j]/
  flags[j]` for arbitrary `j` outside the goroutine's own chunk** (`nndescent.go:289-292`). Two
  goroutines updating each other's neighbor lists race on the same heap slices (`maxheap.go:253/263`).
- Compounding bug: the update counter `updateCounts[workerID]` with `workerID := i*numWorkers/n`
  (`nndescent.go:238,252,262`) is itself a racy write — `workerID` is derived from the loop index, not
  the actual goroutine, so multiple goroutines collide on the same counter slot.

**Why SGD is *not* affected:** `layout.OptimizeLayout` deliberately runs single-threaded whenever a seed
is set — `useParallel := config.Seed == 0 && config.NumWorkers != 1` (`optimize.go:289`). So the layout
stage is reproducible; only the NN stage is broken. The fix is to make NN behave the same way (§6).

**Cross-platform caveat (lower priority):** `float64ToInt64Bits` uses `unsafe.Pointer`
(`optimize.go:~374`) — a little-endian assumption (fine on amd64/arm64; `math.Float64bits` is the
portable replacement). And once the hand-rolled math is replaced with stdlib, "reproducible" means
*same binary + same seed*, not bit-identical coordinates across Go-version/arch upgrades. Set that
expectation in CI and golden tests (§6).

---

## 4. Numerical fidelity analysis

**Good news — the parts that matter are faithful:**
- `graph/smoothKNNDist` mirrors Python's `smooth_knn_dist` (binary search, `local_connectivity`,
  self-skip) using **stdlib `math`** (`fuzzy.go`).
- `layout` gradient terms, ±4 clip, LR decay, negative sampling — all match the reference.
- `internal/rand` reproduces both NumPy's MT19937 and UMAP's Tausworthe RNG (genuinely careful work).

**Measured against real `umap-learn`** (the bundled `python_comparison_test.go`, which we ran live via
`uv`): pairwise-distance **correlation 0.648**, and both implementations separate the clusters. So:
structurally correct, **not** numerically identical — and that's fine for a viz where absolute distances
are meaningless. **Do not chase 0.648 → 0.99.**

**The fidelity risk worth fixing (§6 step 2):** the hand-rolled `fastExp/fastLog/fastPow` in
`layout/optimize.go:226-273` and `exp32/log32/pow32` at `optimize.go:~547`. The `log` is an arctanh
series with a fixed term cap that **loses accuracy for large arguments** — precisely the large embedding
distances seen mid-optimization. On synthetic blobs this is invisible; on real structured data it can
quietly degrade separation. It's also slower than stdlib. The repo *already contains* the correct
replacement (`internal/math`, unused). This likely contributes to the 0.648-not-higher gap, though that
gap is not itself a problem.

---

## 5. Dead / orphaned code inventory

Four distinct pockets. None is dangerous, but the engineer should know what's load-bearing vs. inert
before lifting or modifying. (Commit `0698a1d` already removed a *fifth* pocket — four unused funcs that
were failing lint.)

| Code | Location | Status | Recommendation |
|---|---|---|---|
| RP-tree forest (~360 LOC) | `nn/rptree.go` (all `RPForest` refs are internal) | **Orphaned** — present but never wired into `NNDescent` | Keep; **wire in** for full-census scale (§6 deferred), don't delete |
| `internal/math` (whole pkg) | `internal/math/math32.go` | **Unused** (0 importers) | **Use it** — it's the §6 step-2 replacement for the hand-rolled math |
| Gradient-distance machinery | `distance.go:64-95` (`GradRegistry`, `GetGrad`, `GradFunc`) | **Unused** — layout hardcodes the euclidean output gradient | Leave or trim in a cleanup pass; harmless |
| `NormFloat64` + its math | `internal/rand/tau.go:79-138` | **Unused** — only self-references | Delete in cleanup; removes a 3rd copy of hand-rolled sqrt/log/cos |

---

## 6. What to improve — critical path to v1

Ordered. **Only items 1–4 + the guard are needed for v1.** Effort estimates assume one engineer
familiar with Go.

**1. Fix the NN-descent race → make `nn/nndescent.go` single-threaded. (~0.5–1.5d — gates v1.)**
Don't attempt parallel-deterministic code; it's hard to verify and we don't need the speed (single
-threaded NN is seconds-to-minutes at v1 scale, and this is offline batch). Drop the parallelism in
`initializeRandomNeighbors` and `nnDescentUpdate` (or short-circuit `numWorkers` to 1 inside `NNDescent`).
Do it **unconditionally**, not just "when seeded," so you can turn on `go test -race` in CI as a standing
guard. Re-introduce parallelism later as one unit with the RP-tree work (deferred). Acceptance: the
`n=1200` two-run determinism check passes and `-race` is clean.

**2. Replace hand-rolled math with `internal/math`/stdlib. (~0.5d — do BEFORE item 3.)**
Targets: `layout/optimize.go:226-273` and `:~547-616`; optionally `internal/rand/tau.go:93-138`. Do this
before the real-data test so it isn't judging a known confound. **Acceptance check:** assert
`findABParams(0.1, 1.0)` ≈ Python's `(a≈1.577, b≈0.895)` — a 3-line unit test that proves the swap
improved fidelity (the synthetic separation ratio won't move, so you need this assertion as the signal).

**3. Real-data validation at true v1 scale. (~1d — the actual go/no-go evidence.)**
Not yet done: our 12× separation ratio was synthetic data on the *brute-force + spectral* path. Real v1
(tens of thousands of rows, 15–25 dims) exercises the *random-init + NN-descent* path, which **nothing
has tested.** Run on the real FMCSA matrix at true row count and dimensionality. Use the ground-truth
labels you already have (`is_broker`, equipment archetype) to compute a **labeled separation score**
(not a subjective eyeball), and check **per axis** (brokers-vs-carriers *and* reefer-vs-flatbed — a
global score can pass while one axis collapses). Run it **twice** to confirm reproducibility holds at
scale post-item-1.

**4. Golden regression test to freeze the good state. (~1d.)**
Once 1–3 pass, lock it with a golden test on a **small committed real-data fixture**, asserting a
**tolerance band** (or a separation-metric band) — **not** exact float equality, which is flaky across
Go versions/architectures. Frame it as regression insurance, not a quality target.

**5. Runtime degenerate-output guard. (~10 lines, high value.)**
In the ingest wrapper, after `Fit`, assert the embedding is **all-finite and has spread above a floor**;
**fail the batch loudly** otherwise. This converts the worst failure mode — a silent NaN / collapsed
cloud reaching the renderer — into a hard stop, and protects *every future run*, not just the one you
eyeballed. (The unit tests already check finiteness; promote it to a runtime postcondition.)

**Deferred — explicitly NOT v1** (promote only if item 3 comes back degraded, or full-census scale is
actually needed):
- **Wire in the RP-tree forest** (`nn/rptree.go`) + re-parallelize NN as one unit — improves approximate
  -NN recall at full-census scale.
- **Sparse spectral init** above `n=5000` (`init/spectral.go:57`) — today silently random; fine for v1.
- **Proper out-of-sample `Transform`** — only if daily ingest must project new rows without a full
  recompute. At v1 scale, just recompute the batch. **Keep the staged k-NN `Transform` out of ingest**
  (it's a baseline library method wired into nothing — see below).

**Trivial cleanup (batch whenever):** README/validated-envelope note; the `unsafe` cast
(`optimize.go` → `math.Float64bits`); the biased `rand.Intn` (`tau.go:57`); delete the unused
`NormFloat64` and `GradRegistry`.

**Already done (commit `0698a1d`, cleanup only — does NOT touch SGD numerics):** removed 4 lint-failing
unused funcs (lint now clean); replaced the **previously-panicking** `Transform` (it called a
`getData()→nil` placeholder) with a documented inverse-distance-weighted k-NN projection
(`umap.go`, `transformPoint`); added `nn.NearestK` (`nndescent.go`).

---

## 7. Is lifting it into another repo a simple lift?

**Yes — about as clean as a lift gets**, *because* of the §2 facts: one self-contained package tree, a
single external dependency (`gonum`) confined to one file (`init/spectral.go`), and **no IO, no network,
no global mutable state** — it's pure compute. But it is **"copy the package tree + rewrite the import
prefix," not "copy three files."** Budget **~30–60 min**, low risk.

### File manifest — what to take
| Take | Path | Why |
|---|---|---|
| ✅ | `umap.go` | top-level API (`Fit`/`FitTransform`/`Transform`) |
| ✅ | `distance/` | metrics (stage 1) |
| ✅ | `graph/` | fuzzy simplicial set (stage 2) |
| ✅ | `init/` | initialization (stage 3) — **pulls `gonum`** |
| ✅ | `layout/` | SGD optimizer (stage 4) |
| ✅ | `nn/` | k-NN incl. the orphaned `rptree.go` |
| ✅ | `internal/{heap,math,parallel,rand}/` | helpers (keep `math` — it's the §6 step-2 target) |
| ✅ | `*_test.go`, `testdata/test_data.csv` | your safety net |
| ❌ | `cmd/umap/` | CLI — drop unless you want the binary |
| ❌ | `testdata/python/` | only used by `python_comparison_test.go` (needs `uv`) |

### The four gotchas — all mechanical
1. **Preserve the directory structure.** Go enforces that `internal/` packages are importable only by
   code under the same parent directory. Copy the tree intact; **do not flatten files**, or `nn`/`graph`/
   `layout` will fail to import `internal/*`.
2. **Rewrite the import prefix.** Every `github.com/nozzle/umap/...` → `<your-module>/<dest>/...`. One
   find-and-replace across the copied files. It will not compile until done.
3. **Add the dependency:** `go get gonum.org/v1/gonum@v0.14.0`. *Only* needed for spectral init — if you
   drop `init/spectral.go`'s spectral path (v1 is on random init above n=5000 anyway), the library
   becomes **zero-external-dependency**.
4. **Target repo needs Go 1.25** (the code uses 1.25 language features: range-over-int, builtin
   `min`/`max`, `wg.Go`).

### Recommended approach (don't copy-paste)
Because we still owe the changes in §6, the cleanest path is **to consume it as a module, not vendor
source**: keep this fork as its own module, land items 1–2 + the guard *here*, **cut a tagged release**,
then `go get github.com/<your-fork>/umap@<tag>` from the other repo. Benefits: zero import-rewriting, one
source of truth for the hardening work, and the consumer pins a real version instead of a commit
pseudo-version. **Tag the release *after* item 1 (the race fix)** — tagging today pins the race into your
first "stable" version. Only copy the tree in-repo if your target is a monorepo that forbids external
module deps, in which case follow the four steps above.

---

## 8. Appendix

### `umap.Config` (from `DefaultConfig`, `umap.go`)
| Field | Default | Notes |
|---|---|---|
| `NNeighbors` | 15 | k for the k-NN graph; larger = more global structure, slower |
| `NComponents` | 2 | set to **3** for our cloud |
| `Metric` | `"euclidean"` | affects stage 1 only; see §2 |
| `MinDist` | 0.1 | smaller = tighter clusters |
| `Spread` | 1.0 | with `MinDist`, controls clumpiness |
| `NEpochs` | 200 | 0 ⇒ auto (500 if `n<10000`, else 200) |
| `LearningRate` | 1.0 | SGD α |
| `NegativeSampleRate` | 5 | repulsive samples per positive |
| `Init` | `"spectral"` | **silently → random for `n>5000`** |
| `Seed` | 42 | reproducibility; forces single-threaded SGD |
| `NumWorkers` | 0 | 0 ⇒ GOMAXPROCS (does **not** currently make NN safe — §3) |

### Metrics available (`distance/distance.go:28`)
euclidean/l2, sqeuclidean, manhattan/l1/taxicab, chebyshev/linf, minkowski, **cosine**, correlation,
canberra, braycurtis, haversine, hellinger, and binary: hamming, jaccard, dice, matching, kulsinski,
rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule.

### Commands
```bash
go build ./...                       # compile
go test ./...                        # full suite (python_comparison_test needs `uv` on PATH; auto-skips)
go test -race ./nn/...               # reproduces the NN-descent race (§3, item 1)
go vet ./... && golangci-lint run    # clean as of 0698a1d
go run ./cmd/umap --help             # CLI flags (read cmd/umap/main.go)
```

### Key file:line references
- NN-descent race: `nn/nndescent.go:237-266`, `:289-292`; heap `internal/heap/maxheap.go:253,263`
- SGD single-thread-when-seeded: `layout/optimize.go:289`
- Hand-rolled math to replace: `layout/optimize.go:226-273`, `:~547-616`; `internal/rand/tau.go:93-138`
- Curve fit (a/b): `layout/optimize.go:68` (`findABParams`)
- Spectral → random fallback: `init/spectral.go:57-60`
- Smooth-kNN fidelity (good): `graph/fuzzy.go:142`
- Map-based symmetrization (scale watch): `graph/fuzzy.go:297`
- Brute-vs-NNDescent threshold: `umap.go` (`buildKNNGraph`, `n < 1000`)
- Unused `internal/math`: `internal/math/math32.go`
