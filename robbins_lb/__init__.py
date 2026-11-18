"""robbins_lb -- certified lower bounds for Robbins' problem by a Markov-minorant relaxation.

For a design (Definition 3.3 of the paper) the package evaluates the relaxed Bellman recursion of
Corollary 3.8 backwards, layer by layer:
    v_n >= U_0(sigma_0)   (Lemma 3.1 with the relaxed ranks of Lemma 3.4, or Lemma 3.5 for capped designs),
with the closed-form cell integrals of Section 4.1.

Modules
  design    Design (n, k, fine edges, coarse edges, optional caps); state counts and memory estimates.
  encoding  storage layout of a layer: memory multisets K in blocks by number of entries, count vectors
            ranked lexicographically.
  kernel    numba layer kernel; rounding modes "nearest" and "down" (see its module docstring).
  solver    solve(design, mode, storage) -> dict(U0=..., ...).
  exact     exact-rational reference implementation (fractions), used by the tests.
  designs   build(spec) for the parametrisation of Section 5.1, and the two certified designs D100, D80.

Caps (Lemma 3.5).  Design(caps=(C_1..C_D)) stores the capped cumulative counts min(S_b, C_b) instead of
S_b = #{past observations <= E_{b-1}}.  The capped summary still evolves by a deterministic recursion,
because min(min(S, C) + d, C) = min(S + d, C) for d in {0, 1}, and the capped relaxed rank
1 + min(S_b, C_b) <= 1 + S_b is still a pathwise minorant of the relative rank, so Lemma 3.1 applies.
(A cap C_1 >= k keeps the number of entries of the memory equal to min(k, capped S_1).)

Values returned by mode="nearest" are floating-point approximations of U_0.  Values returned by
mode="down" are rigorous lower bounds of U_0, under the premises of Proposition 4.6 (IEEE-754 binary64
arithmetic with round-to-nearest, and a correct implementation of the summary transitions and relaxed ranks).
"""
from .design import Design, geo_edges, dyadic
from .solver import solve, set_below_normal_priority

__all__ = ["Design", "geo_edges", "dyadic", "solve", "set_below_normal_priority"]
