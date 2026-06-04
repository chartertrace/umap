# UMAP (Go) — Engineering Handoff

**Repo:** fork of `github.com/nozzle/umap` (BSD-3-Clause)
**Branch with our work:** `claude/festive-brahmagupta-UQGr9` (commit `0698a1d`)
**Audience:** engineer lifting this into another repo and finishing it for v1.

---

## 1. What this project is

A **pure-Go implementation of UMAP** — a dimensionality-reduction algorithm. You feed it a
high-dimensional numeric matrix (one row per item, ~15–25 columns) and it returns a low-dimensional
embedding (2 or 3 coordinates per row) where similar items land near each other.

**Our use:** the offline reduction step in a carrier/broker "similarity cloud." We build a feature
matrix per USDOT, UMAP reduces it to 3 coordinates, and a WebGL point cloud renders it. It runs
**offline / batch, never on a user request.** A bad layout is visible immediately and the blast
radius is one batch job — which is why an unproven all-Go library is an acceptable bet here.

It's a complete pipeline, not a fragment:

| Package | Role |
|---|---|
| `distance/` | distance metrics (euclidean, manhattan, cosine, correlation, binary set) + gradients |
| `nn/` | k-nearest-neighbor graph (brute force <1000 rows; approximate "NN-descent" ≥1000) |
| `graph/` | fuzzy simplicial set (the UMAP graph, CSR sparse matrix) |
| `init/` | initial layout (spectral via `gonum` eigendecomposition; random fallback) |
| `layout/` | SGD optimization — the core that produces the final coordinates |
| `umap.go` (root) | top-level `Fit` / `FitTransform` / `Transform` API |
| `cmd/umap/` | CLI wrapper (not needed if you call it as a library) |
| `internal/*` | helpers: `rand` (NumPy-compatible RNG), `heap`, `parallel`, `math` |

**Entry point:** `umap.New(cfg).FitTransform(data)` → `[][]float32` of shape `n × NComponents`,
in input-row order (so you can re-key each row back to its USDOT).

---

## 2. Current state — validated, with one blocker

We ran a validation gate (build, the bundled Python-comparison test against real `umap-learn`,
the race detector, and a synthetic separation test). **Verdict: the algorithm is correct and
produces well-separated layouts.** Concrete evidence:

- Gradient math (attractive/repulsive terms, ±4 clipping, learning-rate decay, negative sampling)
  **matches the reference Python implementation** — not degenerate.
- Synthetic 4-cluster test: clusters separate cleanly (**inter/intra distance ratio ≈ 12**).
- Reproducible (same seed → byte-identical output) **for n < 1000**.
- Builds on **Go 1.25**; `golangci-lint` is clean after our cleanup (commit `0698a1d`).

**The one real blocker → fix before v1:** at **n ≥ 1000** the approximate NN-descent path runs in
parallel and has a **data race** (`go test -race ./nn/...` proves it). Same seed produces
*different* output, which breaks our reproducibility requirement at real scale (v1 = tens of thousands
of rows). This is a concurrency bug, not an algorithm flaw — it's fixable cheaply (see §3).

What commit `0698a1d` already did (cleanup only, **does not touch the SGD numerics**):
- removed dead code → lint clean;
- replaced a **broken `Transform`** (it previously panicked on any fitted model) with a working
  k-NN-weighted projection — **baseline only, keep it out of ingest** (see §3 deferred list);
- added `nn.NearestK`.

---

## 3. What to improve — critical path to v1

Ordered. Only items 1–4 + the guard are needed for v1; the rest is explicitly deferred.

**1. Fix the NN-descent race — make it single-threaded.**
Don't write parallel-deterministic code (hard to verify, and we don't have the scale to need the
speed — single-threaded is seconds-to-minutes at v1). Drop the parallelism in `nn/nndescent.go`
entirely for now; re-introduce it later with item (6). Doing it unconditionally (not just "when
seeded") lets you turn on `-race` in CI as a permanent guard. **~0.5–1.5 day. This gates v1.**

**2. Replace the hand-rolled `exp/log/pow` in `layout/optimize.go` with stdlib / the existing
`internal/math`.** The current Taylor-series versions are inaccurate for large distances and slower.
Do this **before** the real-data test (item 3) so it isn't judging a known confound. Acceptance check:
assert `findABParams(0.1, 1.0)` ≈ Python's `(a≈1.577, b≈0.895)`. **~0.5 day.**

**3. Run the real-data validation at true v1 scale.** This is the actual go/no-go evidence and it's
*not been done* — our 12× ratio was synthetic data on the *brute-force* path. Real v1 (tens of
thousands of rows, 15–25 dims) runs the *random-init + NN-descent* path, which nothing has tested.
Use the ground-truth labels you already have (`is_broker`, equipment archetype) to compute a
**labeled separation score**, not a subjective eyeball; check per-axis (brokers-vs-carriers *and*
reefer-vs-flatbed). Run it **twice** to confirm reproducibility holds at scale post-fix. **~1 day.**

**4. Add a golden regression test to freeze the good state** once 1–3 pass. Use a **tolerance band**
(or a separation-metric band) on a small committed real-data fixture — *not* exact float equality,
which is flaky across Go versions/architectures. Frame it as regression insurance. **Do NOT chase the
Python-match number up; exact fidelity is irrelevant for a viz where absolute distances are
meaningless.** **~1 day.**

**5. Runtime degenerate-output guard (cheap, high value — ~10 lines).** In the ingest wrapper, after
`Fit`, assert the embedding is all-finite and has spread above a floor; **fail the batch loudly**
otherwise. This converts the worst failure mode (a silent NaN / collapsed cloud reaching the renderer)
into a hard stop, and protects every future run, not just the one you eyeballed.

**Deferred — explicitly NOT v1 (only if real-data test comes back degraded, or full-census scale is
actually needed):**
- **Wire in the RP-tree forest** (`nn/rptree.go`, ~360 lines, currently present but *never connected*)
  — improves approximate-NN recall at full-census scale. Do this together with re-parallelizing NN-descent.
- **Sparse spectral init** above n=5000 (today it silently falls back to random init; fine for v1).
- **Proper out-of-sample `Transform`** — only needed if daily ingest must project new rows without a
  full recompute. At v1 scale, just recompute the batch. Keep the staged k-NN `Transform` **out of ingest**.

**Trivial cleanup (batch whenever):** README caveat (record the validated row-count ceiling + the
"reproducible = same binary + seed" guarantee + "`Transform` is a placeholder, not for ingest"); the
`unsafe` float64→int64 cast; a misleading comment in `rand.Intn`.

---

## 4. Is lifting it into another repo a simple lift?

**Yes — this is about as clean as a lift gets.** The whole library is one self-contained package tree
with **a single external dependency (`gonum`), used in exactly one file** (`init/spectral.go`), and no
network/IO/global state — it's pure compute. But it is **"copy the package tree + rewrite the import
prefix," not "copy three files."** Budget **~30–60 minutes**, low risk.

### Dependency graph (all internal, plus one external)
```
umap (root)  ->  distance, graph, init, layout, nn, internal/rand
nn           ->  distance, internal/{heap, parallel, rand}
graph        ->  internal/parallel
layout       ->  graph, internal/{parallel, rand}
init         ->  graph, internal/rand,  gonum/v1/gonum/mat   <-- only external dep
distance, internal/*  ->  leaves (stdlib only)
```

### What to copy ("the right files")
Everything **except** `cmd/umap/` (the CLI) and `testdata/python/` (only used by the comparison test):
`umap.go`, `distance/`, `graph/`, `init/`, `layout/`, `nn/`, and `internal/{heap,math,parallel,rand}/`.
Keep the `_test.go` files too — they're your safety net.

### The four gotchas (all mechanical)
1. **Preserve the directory structure.** The `internal/` packages are only importable by code under
   the same parent dir (Go enforces this). Copy the tree intact; don't flatten files.
2. **Rewrite the import prefix.** Every `github.com/nozzle/umap/...` import must become
   `<your-module>/<dest>/...`. One find-and-replace. It won't compile until you do.
3. **Add the dependency:** `go get gonum.org/v1/gonum@v0.14.0` in the target repo. *(Note: this is
   ONLY needed for spectral init. v1 uses random init above n=5000 anyway — if you drop
   `init/spectral.go`, the library becomes zero-external-dependency.)*
4. **Target repo needs Go 1.25** (the code uses 1.25 language features).

### Recommended approach
Because we still have changes to make (§3), the cleanest path is **NOT copy-paste**: keep this fork as
its own module, do items 1–2 + the guard here, **cut a tagged release**, and `go get` it from the other
repo. Zero import-rewriting, one source of truth, and the other repo pins a real version instead of a
commit hash. Only copy the tree in-repo if your target is a monorepo that forbids external module deps —
in which case follow the four steps above.

> Either way: **tag the release *after* the race fix (item 1)**, so the first pinnable version is
> already reproducible. Tagging today pins in the race.

---

## Appendix — commands
```bash
go build ./...                       # compile
go test ./...                        # full suite (the Python comparison needs `uv` on PATH; it
                                     #   auto-skips if absent)
go test -race ./nn/...               # reproduces the NN-descent race (item 1)
go vet ./... && golangci-lint run    # clean as of commit 0698a1d
go run ./cmd/umap --help             # CLI flags (read cmd/umap/main.go for the list)
```
