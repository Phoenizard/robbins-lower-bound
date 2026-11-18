"""Checks on the frozen production results (results/*.json)."""
import mpmath as mp

from conftest import BF_TABLE3, BF_VN, load_result


def _br(m):
    r = load_result(f"m{m}")
    lo = {int(n): v for n, v in r["W_lower"].items()}
    hi = {int(n): v for n, v in r["W_upper"].items()}
    return lo, hi


# BF Table 3 entries whose rounding interval [v - 5e-5, v + 5e-5] is DISJOINT from our rigorous enclosure.
# (m, n): sign, where "+" = BF printed too high, "-" = too low.  See README section "Mismatches".
BF_MISMATCH = {(1, 17): "-", (2, 17): "+", (3, 17): "+"}


def test_production_enclosures_vs_bf_table3():
    """All 17 BF Table 3 entries are checked; 14 lie in the rigorous enclosures (to the 5e-5 rounding of the
    printed value), exactly the 3 documented ones do not."""
    found = {}
    for m in range(1, 6):
        lo, hi = _br(m)
        for n, s in BF_TABLE3[m].items():
            v = float(s)
            if v + 5e-5 < lo[n]:
                found[(m, n)] = "-"
            elif v - 5e-5 > hi[n]:
                found[(m, n)] = "+"
    assert found == BF_MISMATCH


def test_policy_simulation_confirms_bf_w17_m3_too_high():
    """Independent of recursion (4.5): an explicit stopping rule simulated under loss (4.4) has
    E[Y] < 1.70695 (lower edge of BF's printed 1.7070) by many standard errors, so W_17^(3) < 1.70695."""
    r = load_result("policysim")
    c = next(c for c in r["cases"] if c["m"] == 3 and c["n"] == 17)
    assert c["pooled_mean"] + 6 * c["pooled_se"] < 1.70695
    lo, hi = _br(3)
    assert lo[17] <= c["pooled_mean"] + 4 * c["pooled_se"]       # consistent with the rigorous enclosure
    for c in r["cases"]:                                          # every rule value is >= the rigorous lower bound
        lo, _ = _br(c["m"])
        assert c["pooled_mean"] + 4 * c["pooled_se"] >= lo[c["n"]]


def test_production_lower_le_upper_and_monotone():
    br = {m: _br(m) for m in range(1, 6)}
    for m in range(1, 6):
        lo, hi = br[m]
        for n in range(1, 18):
            assert lo[n] <= hi[n]
            if n < 17:
                assert lo[n] <= hi[n + 1]          # nondecreasing in n
            if m < 5:
                assert lo[n] <= br[m + 1][1][n]    # nondecreasing in m
        assert lo[1] <= 1.0 <= hi[1]


def test_production_m1_contains_exact():
    lo, hi = _br(1)
    ex = load_result("exact")["W1_dps90"]
    for n in range(1, 18):
        v = float(mp.mpf(ex[str(n)]))
        assert lo[n] <= v <= hi[n]


def test_full_information_values():
    ex = load_result("exact")
    v3 = float(mp.mpf(ex["v3_50digits"]))
    v4 = float(mp.mpf(ex["v4_dps40"]))
    assert abs(float(mp.mpf(ex["v4_dps30"])) - v4) < 1e-12
    lo2, hi2 = _br(2)
    lo3, hi3 = _br(3)
    lo4, hi4 = _br(4)
    assert lo2[3] <= v3 <= hi2[3]                  # v_3 = W_3^(2)
    assert lo3[4] <= v4 <= hi3[4]                  # v_4 = W_4^(3)
    assert lo4[5] - 5e-5 <= float(BF_VN[5]) <= hi4[5] + 5e-5   # BF's V_5 = W_5^(4)
    assert abs(v3 - float(BF_VN[3])) < 1e-4                    # printed 1.3915, true 1.391552


def test_kbest_table1_reproduced():
    r = load_result("kbest")
    assert len(r["cases"]) >= 14
    n_round = 0
    for c in r["cases"]:
        pub = c["published"]
        ulp = 10 ** (-len(pub.split(".")[1]))
        diff = abs(c["code_variant"] - float(pub))
        assert diff <= ulp + 1e-12, c                      # within one unit of the last printed digit
        n_round += diff <= 0.5 * ulp + 1e-12
        # Definition 5 as printed does not reproduce Table 1 at n = 10 (k < n - 1 regime)
        if c["n"] == 10:
            assert abs(c["paper_variant"] - float(pub)) > ulp
    assert n_round >= len(r["cases"]) - 1                  # only v_{4,500,2} looks truncated, not rounded
