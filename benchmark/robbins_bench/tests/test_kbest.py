"""Reproduction of Brice-Bruss-Majumdar-Raskin (arXiv 2608.27419) numbers with the (d,k)-abstraction."""
import pytest

from robbins_bench.kbest import kbest_value


def rounds_to(x, s):
    dec = len(s.split(".")[1])
    return abs(x - float(s)) <= 0.5 * 10 ** (-dec) + 1e-12


@pytest.mark.parametrize("n,d,k,pub", [
    (3, 500, 3, "1.391635988"),     # Sec. 6.1, M_{3,500} (= M_{3,500,3}, Lemma 2)
    (5, 100, 2, "1.57347"),         # Table 1
    (5, 100, 3, "1.57219"),
    (10, 100, 2, "1.80133"),
    (10, 100, 3, "1.79654"),
    (10, 500, 2, "1.79912"),
])
def test_code_variant_reproduces_published(n, d, k, pub):
    assert rounds_to(kbest_value(n, d, k, "code"), pub)


def test_v4_500_2_within_one_unit():
    # we get 1.4943354630; printed 1.49433 (truncated rather than rounded; diff 5.5e-6 < 1e-5)
    v = kbest_value(4, 500, 2, "code")
    assert abs(v - 1.4943354630156864) < 1e-12
    assert abs(v - 1.49433) < 1e-5 and not rounds_to(v, "1.49433")


def test_paper_definition5_does_not_reproduce_table1():
    """Definition 5 / eq. (10) as printed gives different numbers (forgotten-draw count differs)."""
    v = kbest_value(10, 100, 2, "paper")
    assert not rounds_to(v, "1.80133")
    assert abs(v - 1.80389) < 1e-5


def test_no_forgetting_variants_agree():
    # with k >= n the buffer never overflows and both readings coincide (= M_{n,d}, Lemma 2);
    # in the code variant k = n - 1 is already exact (its F term vanishes at the last draw)
    assert abs(kbest_value(4, 60, 4, "code") - kbest_value(4, 60, 4, "paper")) < 1e-14
    assert abs(kbest_value(4, 60, 4, "code") - kbest_value(4, 60, 3, "code")) < 1e-14
