# robbins-lower-bound

Code and certificate records for the paper

> Xiaoyi Jiang, *A certified lower bound for Robbins' problem of minimizing the expected rank*,
> [CITATION / arXiv identifier to be added].

Authors: Xiaoyi Jiang. License: MIT (see `LICENSE`).

## Results and the original certificates

In Robbins' problem one observes i.i.d. uniform random variables X_1, ..., X_n one at a time. The aim is
to stop, using only the values seen so far, so as to minimize the expected absolute rank of the selected
observation among all n. The optimal value v_n is nondecreasing in n, and its limit V is unknown
(Bruss and Ferguson 1993, Theorem 2).

**Theorem 6.1 of the paper.** V ≥ v_100 ≥ 2.0968.

The certified run of the design D100 returned the binary64 number

    L_0 = 4397361 / 2^21 = 2.096825122833251953125   (0x1.0c64c40000000p+1),

and L_0 ≤ v_100. D100 uses the caps of Lemma 3.5.

**Corollary 6.2 of the paper (fallback, without caps).** V ≥ v_80 ≥ 2.0513. The certified run of the
uncapped design D80 returned L_0 = 8603877 / 2^22 = 2.0513241291046142578125 (0x1.0691ca0000000p+1). This
bound does not use Lemma 3.5.

For comparison, the paper reports V > 1.85 (Assaf and Samuel-Cahn 1996) as the best previously proved lower
bound it is aware of.

## How the code relates to the paper

The proof replaces the relative rank by a pathwise minorant (the *relaxed rank*). The minorant depends on
the history only through a small recursive *summary*: the fine-grid positions of the k smallest
observations below a threshold θ, and (possibly capped) cumulative counts of observations below a few
coarse levels. A *design* fixes k, θ, the fine and coarse edges and the caps. The value U_0 of the
resulting finite Markov decision problem is a lower bound on v_n. The code evaluates U_0 from below in
floating point with sign-aware directed rounding.

| paper | code |
|---|---|
| Definition 3.3 (design), Section 5.1 (parametrisation, caps rule) | `robbins_lb/design.py` (`Design`, `Design.hybrid`), `robbins_lb/designs.py` (`build`, `D100`, `D80`) |
| Summary σ_t = (K_t, Ŝ(t)), relaxed rank eq. (5), Lemmas 3.4 and 3.5 | `robbins_lb/encoding.py` (memory tables, count vectors), successor computation in `robbins_lb/kernel.py::layer_kernel` |
| Corollary 3.8, eqs. (7)–(8) (relaxed recursion) | `robbins_lb/solver.py::solve` (backward pass), `robbins_lb/kernel.py::layer_kernel` (one layer) |
| Cell integral eq. (9), Lemma 4.1 | `robbins_lb/kernel.py::panel_near` |
| Definition 4.3, Lemma 4.4 (rounded-down cell integral) | `robbins_lb/kernel.py::panel_down`, `dn`, `up` |
| Section 4.3, Proposition 4.6 (certified backward pass, binary32 store rounded down) | `layer_kernel` with `down=True`, `closed_form_A`, `closed_form_B` |
| Section 5.4 (certificate record) | `scripts/certify.py` |
| Exact-rational evaluation of small designs (Appendix C) | `robbins_lb/exact.py`, `tests/` |
| Section 6.2, item 1 (second implementation) | `replication/` |
| Appendix C.4 (enclosures of W_n^(m)), exact v_1..v_3 and a high-precision v_4 reference | `benchmark/` |
| Table 1, columns "MC dual" and "MC rule" (not certified) | `supplementary/mc_dual/` |

Lemma, equation and section numbers refer to the current version of the manuscript.

## Premises of this computer-assisted proof

Theorem 6.1 holds if all of the following hold.

1. **The mathematics.** The paper cites Theorem 2 of Bruss and Ferguson for monotonicity and proves the remaining steps:
   - Lemma 2.1;
   - Lemma 3.1, Lemma 3.4 and Lemma 3.5 (for D100), and Corollary 3.8;
   - Lemmas 4.1, 4.2 and 4.4, and Proposition 4.6.
2. **The implementation.** The code implements the summary transitions Φ and the relaxed ranks r̂ of
   Section 3.
3. **The arithmetic as executed.** Every binary64 operation is correctly rounded to nearest (eq. (10)).
   The binary32 store rounds down as described in Section 4.3. Section 4.4 checks this for numba 0.68.1
   with llvmlite 0.50.0 on x86-64:
   - no fast-math flags, no contraction, no fused multiply-add;
   - `dn` and `up` are exact bit-pattern steps;
   - the binary32 `nextafter` lowers to the single-precision routine.

   Other versions of numba or llvmlite have not been checked; `requirements.txt` pins these two.
4. **The run itself.** The run executes the committed code:
   - with a clean checkout, a fresh compilation cache and a fresh checkpoint directory;
   - on hardware and an operating system that execute it correctly.

   `scripts/certify.py` checks the clean checkout and fresh directories at startup and records the code version.
   Its JSON flag `certified` indicates use of down mode. Applying Proposition 4.6 also requires the
   implementation and arithmetic premises above.

**Important.** The two certificate records in `certificates/original/` were produced from the earlier
layout of the research repository, not from this repository. The certified core here is the same code
with rewritten docstrings: the core modules are AST-identical to the certified ones
(docstrings removed). The records apply to the research commits named in
`certificates/original/README.md`. Anyone can re-certify from this repository with `scripts/certify.py`
(about 1 h for D100 on an 8-thread laptop, peak about 3.7 GiB); the expected result is the exact value above.

## Checks done

All checks below are numerical evidence for premises 2 and 3. None of them is part of the proof.

- **Unit tests** (`tests/`, 80 tests):
  - the encoding against brute-force enumeration;
  - round-to-nearest mode against the exact-rational module (< 1e-13), and down mode ≤ the exact
    rational value, compared exactly, with binary64 and binary32 storage;
  - U_0 ≤ v_n for n ≤ 4 against exact references through n = 3 and a high-precision reference at n = 4;
  - monotonicity under refinement of the edges, increase of k and increase of the caps;
  - 2·10^5 adversarial cells, down mode ≤ the exact value in every case;
  - `dn`/`up` against `nextafter` on 10^6 values and all special values;
  - checkpoint/resume;
  - `D100` and `D80` equal the certificate designs bit for bit, with identical state counts;
  - `benchmark/`, `replication/` and `supplementary/` do not import `robbins_lb`.
- **Independent second implementation** (`replication/`). It was written from the specification only. It
  reproduces the certified D100 value 4397361/2^21 bit for bit and has the same number of summaries per
  layer. See `replication/README.md`.
- **Independent benchmark** (`benchmark/`): exact v_1..v_3, a high-precision v_4 reference, and enclosures of
  the Bruss–Ferguson W_n^(m).
- **Line-by-line review of the rounding** in the down path, and of the compiled code for the toolchain
  above (Section 6.2, item 3, and Section 4.4 of the paper).
- **Regression of this repository against the research code.**
  - The core modules are AST-identical modulo docstrings.
  - On the n = 30 capped design of `scripts/quick_check.py`, the results of this repository are
    bit-identical to those of the research code:

    | mode | storage | this repository | research code |
    |---|---|---|---|
    | down | f64 | 1.9727237588595965 | 1.9727237588595965 |
    | nearest | f32 | 1.9727237224578857 | 1.9727237224578857 |

  - The original certificate records pass `scripts/verify_certificate.py --git-repo` against the research
    repository.

## Reproduce

**Environment.** CPython 3.12 and the pinned packages:

```
python -m pip install -r requirements.txt
```

The reported runs used Python 3.12.16 on Windows 11 x64, on a laptop with an Intel Core i7-12700H and
16 GB of memory, with at most 8 threads at below-normal priority. All computations are deterministic, and
the results do not depend on the number of threads.

All commands are run from the repository root. Each run should use a new, empty numba cache directory, so
that the kernels are compiled from the current source.

| step | command | time (reference machine) | memory | expected output |
|---|---|---|---|---|
| unit tests | `python -m pytest tests -q` | ~3.5 min (4 threads) | < 1 GB | `80 passed` |
| quick check | `python scripts/quick_check.py --threads 4 --also-nearest-f32` | 5–12 min at 4 threads (down/f64: 283 s and 669 s in two runs; nearest/f32 adds ~70 s) | 2.5 GiB for the two live layers (f64), ~3.6 GB committed | `QUICK CHECK PASSED` |
| original certificate files | from `certificates/original/`, run `sha256sum -c SHA256SUMS` | seconds | – | four matching hashes (see `certificates/original/README.md` for source verification and PowerShell) |
| fallback D80 | `python scripts/certify.py n80_cert D80 --mode down --storage f32` | ~8 min (8 threads) | 4.2 GiB peak | `L_0 = 8603877/4194304` |
| main D100 | `python scripts/certify.py n100_cert D100 --mode down --storage f32` | ~1 h (8 threads) | 3.7 GiB peak | `L_0 = 4397361/2097152` |
| benchmark | see `benchmark/README.md` | ~70 min + tests 45 s | ≤ 2.8 GB | 56 tests pass |
| replication | `cd replication && python -X utf8 rep_validate.py` | ~150 s | < 2 GB | all checks pass |

`scripts/certify.py` requires `NUMBA_CACHE_DIR` to be set to an existing, empty directory, a clean git
working tree and a fresh `runs/RUNID/`. In bash:

```
mkdir /tmp/nbc_n100 && NUMBA_CACHE_DIR=/tmp/nbc_n100 python scripts/certify.py n100_cert D100 --mode down --storage f32
```

In PowerShell:

```
New-Item -ItemType Directory C:\tmp\nbc_n100; $env:NUMBA_CACHE_DIR = "C:\tmp\nbc_n100"
python scripts/certify.py n100_cert D100 --mode down --storage f32
```

The driver writes `runs/RUNID/RUNID.cert.json` and `runs/RUNID/RUNID.log`. Check a new record with
`python scripts/verify_certificate.py runs/RUNID/RUNID.cert.json`. The D100 and D80 values in the table
are those of the original certified runs; this repository has not yet been used for a certified run of
these two designs.

## Repository map

```
README.md                      this file
LICENSE                        MIT License
requirements.txt               pinned versions
robbins_lb/                    the certified core
  design.py                    Design: edges, caps, state counts, memory estimate
  encoding.py                  layer storage layout (memory multisets, count vectors)
  kernel.py                    numba layer kernel; cell integrals in "nearest" and "down" mode
  solver.py                    solve(design, mode, storage): backward pass with checkpoints
  exact.py                     exact-rational reference (tests)
  designs.py                   build(spec); D100, D80; the quick-check spec
scripts/
  certify.py                   certified-run driver (writes the certificate record)
  quick_check.py               small certified run (n = 30) with its expected value
  verify_certificate.py        recompute the hashes listed in a certificate record
tests/                         unit tests of robbins_lb
benchmark/                     independent validation benchmark (does not import robbins_lb)
replication/                   independent second implementation (does not import robbins_lb)
supplementary/mc_dual/         Monte-Carlo dual bounds; NOT part of the proof
certificates/original/         the two original certificate records and logs (byte-identical)
```

## Citation

[CITATION to be added: authors, title, arXiv identifier, year.]
