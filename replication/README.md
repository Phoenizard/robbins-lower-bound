# Independent second implementation

This directory contains a second implementation of the relaxation (item 1 of Section 6.2 of the paper).
It was written from the written specification only (the statements of Sections 3 and 4 of the paper and
the description of the designs), without reading or importing the main implementation (`robbins_lb`).
It does not import `robbins_lb` and shares no code with it. It is numerical evidence that the main code
implements the summary transitions and relaxed ranks correctly. It is not part of the proof.

## How it differs from the main implementation

- **Encoding.** All capped count vectors are enumerated once and sorted by (s_D, ..., s_1), so layer t is
  a prefix of one global index space. Successor tables are precomputed by binary search on mixed-radix
  codes. Memory multisets are ranked by dictionaries.
- **Out-of-core option.** A layer can be computed slab by slab (slab = fixed s_D), with only three slabs
  in memory (`disk` backend, resumable).
- **Summation order.** Coarse bins D..1 first, then fine cells G..1.
- **Cell integral.** P = y·h − max(0, y − clo)²/(2m) with y = min(w, chi), instead of the case distinction
  of eq. (9) of the paper.
- **Directed rounding.** Each +, −, ×, / is evaluated in round-to-nearest. The sign of its exact error is
  then obtained by an error-free transformation (TwoSum, Dekker's TwoProduct, an exact residual for
  division), and the result is moved by at least one ulp only if it lies on the wrong side. The binary32
  predecessor uses `frexp`; `nextafter` is not used.
- **Exact reference.** `rep_exact.py` uses the full sorted history of cell indices as its state and
  recomputes every rank from it. It does not use the summary, the cap recursion or the form (9).

## Files

| file | content |
|---|---|
| `rep_design.py` | designs: edges θ·l/G and θ + (1−θ)·expm1(βb/D)/expm1(β), caps |
| `rep_exact.py` | exact rational reference over full histories (tiny designs only) |
| `rep_solver.py` | numba backward pass, modes `nearest` and `down`, storage f32/f64, backends `ram` and `disk` |
| `rep_env.py` | thread count and below-normal priority |
| `rep_validate.py` | validation suite (writes `validation_results.json` here) |
| `rep_run.py` | production runs: `python -X utf8 rep_run.py NAME n k th G D beta caps mode storage backend threads` |
| `rep_run_certified.py` | run of the certified design using the edges recorded in `../certificates/original/n100_g20_down_f32.cert.json` |
| `rep_regress.py` | regression against four floating-point values of uncapped designs |

The `disk` backend and `rep_validate.py` put temporary files in the directory named by the environment
variable `REP_WORKDIR` (required for `disk`). Outputs are written to `results/` next to the scripts.

To repeat the D100 replication, from this directory run:

```
python -X utf8 rep_run_certified.py d100_replica down f32 ram 8
```

Compare `U0_exact_fraction` in `results/d100_replica.json` with `4397361/2097152`.

## Results recorded during the original replication

All runs used numba 0.68.1 at below-normal priority. The original D100 machine-readable outputs are in [recorded/](recorded/). They include both modes, the exact edges and caps, and the per-layer state counts.

| design | main implementation | this implementation | agreement |
|---|---|---|---|
| n=30 (k=3, θn=7, G=24, D=5, β=3), caps (12,13,15,18,24); nearest, f32 storage | 1.9727182388305664 | 1.9727182388305664 | bit-identical |
| n=50 (3, 7, 16, 6, 3.5), caps (12,13,15,19,25,36); nearest, f32 storage | 2.042262077331543 | 2.042262077331543 | bit-identical |
| D100 (certified design); down, f32 storage rounded down | 4397361/2^21 (certificate) | 4397361/2^21 = 0x1.0c64c40000000p+1 | bit-identical |
| D100; nearest, f32 storage | 2.096830129623413 | 2.096830129623413 | bit-identical |

Further facts:

- The numbers of summaries per layer agree exactly in all four cases. For example, D100 has
  466,686,002 summaries in its largest layer and 32,183,421,973 in total.
- At n=30 the rounded-down evaluation of this implementation gives 1.9727181759837937 with f64 storage and
  1.9727168083190918 with f32 storage.
- The D100 run took 3730 s at 8 threads with a peak working set of 3.72 GiB (`ram` backend).
- **Edges of D100.** This implementation's own edge formula reproduces the certificate's caps
  (8, 9, 10, 14, 23, 44). It differs from the certificate's binary64 edges by 1 to 3 ulp on four of the
  27 edges (fine edges 11 and 17, coarse edges E_2 and E_4), because the floating-point evaluation
  order differs. So the D100 runs read the certificate's exact binary64 edges as data. This way the same
  mathematical relaxation is evaluated.
- Bit-identity is not guaranteed a priori, since the rounding procedures differ. It occurs because each
  layer's binary32 round-down absorbs the ~1e-15 differences of the binary64 accumulations.

Validation suite (`rep_validate.py`, about 150 s, all checks passed):

- Directed primitives against `Fraction` on 10^5 random pairs: no result on the wrong side, all within
  2 ulp. The binary32 predecessor equals NumPy's `nextafter` on 10^5 values.
- Cell integrals against the exact rational integral on 6·10^4 cells (a third of them with w within
  ±4 ulp of a branch point): the `down` value is never above the exact value.
- Whole solver against the full-history exact reference on 47 designs (n = 2..7, k ≤ 3, random edges,
  caps binding in most): |nearest − exact| ≤ 4.4e-16. `down` is ≤ exact in all 47 designs, by at most
  2.4e-15 with f64 storage and 3.7e-7 with f32 storage.
- U_0 ≤ v_n for n ≤ 4, using the exact values through n = 3 and the high-precision n = 4 reference in `../benchmark/results/exact.json`, with an O(1/G) gap.
  For n = 5, 6, U_0 is below 1.5741707 and 1.6399473, the values of a pessimistic grid rule (Table 3 of
  the paper). These are upper bounds on v_n up to floating-point rounding; the gaps are at least 0.017.
  The pessimistic grid rule is not included in this repository.
- Pure best-k relaxation (k = 1, θ = 1) is ≤ the exact Bruss–Ferguson W_n^(1) for n = 5, 9, 17, and the
  gap shrinks as the grid is refined.
- The `ram` and `disk` backends give bit-identical results.
- 45 monotonicity checks in exact rationals (refining bins or cells, increasing caps or k) found no
  violation.
- Regression against four uncapped floating-point values (n = 10, 20, 30): agreement to the 4 printed
  decimals.

Not re-run here: the binary64-storage runs of D100 (they need about 250 GB of temporary slab I/O with the
`disk` backend) and the rounded-down run of the n=50 design.
