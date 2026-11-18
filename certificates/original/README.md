# Original certificate records

This directory contains the records of the two certified runs reported in the paper, as byte-identical
copies of the files written at the time. They were **not** produced from this repository: they were
produced from the earlier layout of the research repository from which this repository was derived. The
certified core modules of this repository are AST-identical to the ones used (only docstrings differ); both
designs can be re-certified from this repository with `scripts/certify.py`.

| file | SHA-256 |
|---|---|
| `n100_g20_down_f32.cert.json` | `435b208b1c61f23b7558fe412ed81ac55e1812f81b03611f7dc8cd0aa18e7a08` |
| `n100_g20_down_f32.log` | `3ddb85dd343391f41a58cd80b6b72a2c2d9d02ef42b92d450d9c83e36a71b926` |
| `n80_nocap_down_f32.cert.json` | `f946e6c75eb584af754ae2c3dcb1c11b2d1af1ab8e485aac9e922df0ed5fb782` |
| `n80_nocap_down_f32.log` | `9691140c6ba01da31c7fd0105ac47f455d1ca7f24b97a8dd1660530905a4c04e` |

These are the hashes printed in Appendix D of the paper (also in `SHA256SUMS`; check with
`sha256sum -c SHA256SUMS` from this directory). Git stores these four files without line-ending
conversion (`.gitattributes`); the logs have CRLF line endings, as written on Windows.

In PowerShell, from this directory, verify the same manifest with:

```powershell
$failed = $false
Get-Content SHA256SUMS | ForEach-Object {
    $expected, $name = $_ -split '\s+', 2
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $name -ErrorAction Stop).Hash
    if ($actual -ne $expected) {
        $failed = $true
        Write-Error "SHA-256 mismatch for $name"
    } else {
        Write-Output "$name OK"
    }
}
if ($failed) { throw "Certificate file verification failed" }
```

## The two runs

| record | design | result `L_0` | research commit | used in |
|---|---|---|---|---|
| `n100_g20_down_f32` | `D100`: n=100, k=3, θn=7, G=20, D=6, β=5, caps rule [0.9, 0, 1] → (8, 9, 10, 14, 23, 44) | 4397361/2^21 = 2.096825122833251953125 | `803526b83b9a3fa2a4a8da6a03a1e40cc478b803` | Theorem 6.1 |
| `n80_nocap_down_f32` | `D80`: n=80, k=3, θn=7, G=12, D=4, β=3, no caps | 8603877/2^22 = 2.0513241291046142578125 | `6c93475577e1ee7a4b2f86ff7fb6d963ed1aed05` | Corollary 6.2 |

Both runs: mode `down` (sign-aware directed rounding), binary32 storage rounded down, 8 threads, a fresh
and empty `NUMBA_CACHE_DIR`, a fresh checkpoint directory, and a clean source tree. Python 3.12.16,
NumPy 2.5.4, numba 0.68.1, llvmlite 0.50.0, Windows 11 on an Intel Core i7-12700H with 16 GB of memory.
The records contain absolute paths of the original machine (`numba_cache_dir`, `ckpt_dir`); they are kept
unchanged because the files are archived byte for byte.

## Which file maps to which

The records list the SHA-256 of every source file that was loaded, under its path in the research
repository. The two commits have byte-identical copies of all eight files. Their counterparts here:

| path in the research repository (as recorded) | file in this repository | relation |
|---|---|---|
| `src/robbins/relax/design.py` | `robbins_lb/design.py` | same code; docstrings rewritten |
| `src/robbins/relax/encoding.py` | `robbins_lb/encoding.py` | same code; docstrings rewritten |
| `src/robbins/relax/kernel.py` | `robbins_lb/kernel.py` | same code; docstrings rewritten |
| `src/robbins/relax/solver.py` | `robbins_lb/solver.py` | same code; docstrings rewritten |
| `src/robbins/relax/__init__.py` | `robbins_lb/__init__.py` | same code; docstring rewritten |
| `src/robbins/__init__.py` | (none) | docstring-only package file of the research repository |
| `exp/P1/search.py` | `robbins_lb/designs.py` | the function `build(spec)`, unchanged; the search loop is not included |
| `exp/P2/certify.py` | `scripts/certify.py` | rewritten driver with the same checks plus a whole-tree clean check |

"Same code" means that the abstract syntax trees, with docstrings removed, are identical. `robbins_lb/exact.py` is not listed in the records
because the certified run does not load it.

Because of the docstring changes and the line-ending normalisation, the hashes of the files in this
repository differ from the recorded ones; that is why `scripts/verify_certificate.py` reports MISSING files
for these two records. Anyone with access to the research repository can check the recorded hashes against
its commits:

```
python scripts/verify_certificate.py certificates/original/n100_g20_down_f32.cert.json --git-repo PATH_TO_RESEARCH_REPO
python scripts/verify_certificate.py certificates/original/n80_nocap_down_f32.cert.json --git-repo PATH_TO_RESEARCH_REPO
```

Both commands print `CERTIFICATE FILES VERIFIED`. The hashes were taken from the bytes of the working tree
on the machine that ran the computation, while git stores LF-normalised blobs. Three files
(`design.py`, `kernel.py`, `certify.py`) had CRLF line endings in that working tree and five had LF, so the
script accepts either line-ending variant of each blob and reports which one matched.
