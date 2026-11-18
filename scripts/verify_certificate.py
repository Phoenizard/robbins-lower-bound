"""Check a certificate record written by scripts/certify.py.

Usage:  python scripts/verify_certificate.py CERT.json [--root DIR | --git-repo DIR]

1. Recomputes the SHA-256 of every file listed in the record's "sha256" field, relative to DIR (default: the
   root of this repository), and compares it with the recorded hash.
2. Checks that the recorded result is internally consistent: U0_repr, U0_hex, U0_exact_fraction and
   U0_exact_decimal must all denote the same binary64 number.
3. Prints the recorded commit; if DIR is a git checkout, also says whether HEAD equals it.
With --git-repo DIR, step 1 instead reads each file from the recorded commit of the git repository DIR
(`git show COMMIT:PATH`) and accepts it if the blob matches the recorded hash with LF or with CRLF line endings
(git stores LF-normalised blobs, while the hashes were taken from the bytes of the working tree).

Exit status 0 iff every listed file exists and matches and the result fields are consistent.

Note: the two records in certificates/original/ list files of the earlier research layout
(src/robbins/relax/*.py, exp/P1/search.py, exp/P2/certify.py), hashed from the bytes of the working tree on
the machine that ran them (three files with CRLF, five with LF line endings).  Against this repository they report MISSING; they can be checked with --git-repo against the
research repository (see certificates/original/README.md).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cert")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--git-repo", default=None)
    a = ap.parse_args()
    rec = json.load(open(a.cert))
    ok = True

    print(f"record: {a.cert}")
    print(f"runid {rec.get('runid')}  mode {rec.get('mode')}  storage {rec.get('storage')}  "
          f"commit {rec.get('git_commit')}")
    for rel, h in rec["sha256"].items():
        if a.git_repo:
            try:
                blob = subprocess.check_output(["git", "-C", a.git_repo, "show", f"{rec['git_commit']}:{rel}"],
                                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f"  MISSING   {rel} (not in commit {rec['git_commit'][:12]})")
                ok = False
                continue
            lf = blob.replace(b"\r\n", b"\n")
            variants = {"LF": hashlib.sha256(lf).hexdigest(),
                        "CRLF": hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}
            hit = [k for k, v in variants.items() if v == h]
            ok &= bool(hit)
            print(f"  {'OK      ' if hit else 'MISMATCH'}  {rel}  {h[:16]}  ({hit[0] + ' line endings' if hit else 'no variant matches'})")
            continue
        p = os.path.join(a.root, rel)
        if not os.path.exists(p):
            print(f"  MISSING   {rel}")
            ok = False
            continue
        got = sha(p)
        same = got == h
        ok &= same
        print(f"  {'OK      ' if same else 'MISMATCH'}  {rel}  {got[:16]}{'' if same else '  (recorded ' + h[:16] + ')'}")

    u = float(rec["U0_repr"])
    num, den = (int(x) for x in rec["U0_exact_fraction"].split("/"))
    fr = Fraction(num, den)
    cons = (float.fromhex(rec["U0_hex"]) == u and Fraction(u) == fr
            and Fraction(Decimal(rec["U0_exact_decimal"])) == fr)
    ok &= cons
    print(f"  result fields {'consistent' if cons else 'INCONSISTENT'}: U0 = {rec['U0_exact_fraction']} "
          f"= {rec['U0_exact_decimal']} = {rec['U0_hex']}")

    try:
        head = subprocess.check_output(["git", "-C", a.git_repo or a.root, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        print(f"  HEAD of {a.git_repo or a.root} is {head}: {'equal to' if head == rec.get('git_commit') else 'different from'} "
              f"the recorded commit")
    except Exception:
        pass
    print("CERTIFICATE FILES VERIFIED" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
