"""Storage layout of the summaries of the hybrid relaxation (Appendix B.2 of the paper).

Layer t (summaries after t observations) is stored as one flat array indexed by

    idx(K, S) = base[S_1] + (sid(S) - first[S_1]) * bs[j] + loc(K),     j = min(k, S_1),

where
  * S = (S_1..S_D) is the nondecreasing count vector with S_b <= ub_b(t) (ub = t, or min(t, C_b) if capped);
    sid(S) is its rank in lexicographic order (`rank_seq`); sequences with the same S_1 = v occupy the
    contiguous rank range [first[v], first[v+1]);
  * K is a memory multiset; only multisets with exactly j = min(k, S_1) real entries are reachable together
    with S_1 (each observation <= theta enters K until K is full), so each S gets a block of bs[j] = C(Gf+j-1, j)
    memory states and loc(K) is the index of K inside its block.

Memory multisets: sorted tuples (a_1 <= ... <= a_k) over {0..Gf-1} U {EMPTY = Gf}; the global memory id `kid`
lists block 0, then block 1, ..., block k; inside a block, tuples are in lexicographic order.
"""
from __future__ import annotations

import itertools
import math

import numba as nb
import numpy as np


# ----------------------------------------------------------------------------- memory multisets
def memory_tables(k: int, Gf: int):
    """Returns dict with
       nk          total number of memory ids;
       bstart[j]   first global id of block j (j = 0..k), bstart[k+1] = nk;
       bs[j]       block sizes;
       block[g]    number of real entries of memory g;
       loc[g]      index of g inside its block;
       nxt[g, l]   global id after inserting fine cell l (0-based) and keeping the k smallest;
       nxtloc[g,l] its local index;
       cnt[g, l]   #{q in K_g : q < l}  (fine rank minorant minus 1);
       tuples      list of the tuples (for tests / debugging)."""
    EMPTY = Gf
    tuples = []
    bstart = []
    for j in range(k + 1):
        bstart.append(len(tuples))
        for c in itertools.combinations_with_replacement(range(Gf), j):
            tuples.append(tuple(c) + (EMPTY,) * (k - j))
    nk = len(tuples)
    bstart.append(nk)
    index = {tp: g for g, tp in enumerate(tuples)}
    block = np.zeros(nk, dtype=np.int64)
    loc = np.zeros(nk, dtype=np.int64)
    for j in range(k + 1):
        for g in range(bstart[j], bstart[j + 1]):
            block[g] = j
            loc[g] = g - bstart[j]
    nxt = np.zeros((nk, Gf), dtype=np.int64)
    cnt = np.zeros((nk, Gf), dtype=np.int64)
    for g, tp in enumerate(tuples):
        for l in range(Gf):
            cnt[g, l] = sum(1 for q in tp if q < l)
            if k == 0:
                nxt[g, l] = 0
            else:
                new = tuple(sorted(tp + (l,))[:k])
                nxt[g, l] = index[new]
    nxtloc = loc[nxt]
    bs = np.array([bstart[j + 1] - bstart[j] for j in range(k + 1)], dtype=np.int64)
    for j in range(k + 1):
        assert bs[j] == math.comb(Gf + j - 1, j) if j > 0 else bs[j] == 1
    return dict(nk=nk, bstart=np.array(bstart, dtype=np.int64), bs=bs, block=block, loc=loc,
                nxt=nxt, nxtloc=nxtloc, cnt=cnt, tuples=tuples)


# ----------------------------------------------------------------------------- count sequences
def count_table(ub):
    """cnt[p, v] = number of nondecreasing (S_p..S_{D-1}) (0-based p) with S_p = v and S_q <= ub[q].
    ub must be nondecreasing.  Shape (D, ub[-1] + 2) (last column zero, as a sentinel)."""
    D = len(ub)
    W = int(ub[-1]) + 2
    cnt = np.zeros((D, W), dtype=np.int64)
    for v in range(int(ub[D - 1]) + 1):
        cnt[D - 1, v] = 1
    for p in range(D - 2, -1, -1):
        # suffix sums of cnt[p+1]
        suf = np.cumsum(cnt[p + 1, ::-1])[::-1]
        for v in range(int(ub[p]) + 1):
            cnt[p, v] = suf[v]
    return cnt


def layer_encoding(ub, k, bs):
    """Arrays describing layer storage: cnt table, pref[p, v] = sum_{x<v} cnt[p, x],
    first[v] = pref[0, v] (first sid with S_1 = v), base[v] (first flat index of the S_1 = v group).
    Returns (cnt, pref, base, nstates)."""
    cnt = count_table(ub)
    pref = np.zeros((cnt.shape[0], cnt.shape[1] + 1), dtype=np.int64)
    pref[:, 1:] = np.cumsum(cnt, axis=1)
    W = cnt.shape[1]
    base = np.zeros(W + 1, dtype=np.int64)
    for v in range(W):
        base[v + 1] = base[v] + cnt[0, v] * bs[min(k, v)]
    return cnt, pref, base, int(base[W])


@nb.njit(cache=True)
def rank_seq(S, pref):
    """lexicographic rank of the nondecreasing sequence S among the valid sequences of the layer."""
    r = 0
    prev = 0
    for p in range(S.shape[0]):
        r += pref[p, S[p]] - pref[p, prev]
        prev = S[p]
    return r


@nb.njit(cache=True)
def unrank_seq(sid, cnt, S):
    rem = sid
    prev = 0
    for p in range(S.shape[0]):
        v = prev
        while rem >= cnt[p, v]:
            rem -= cnt[p, v]
            v += 1
        S[p] = v
        prev = v


@nb.njit(cache=True)
def next_seq(S, ub):
    """lexicographic successor in place; returns False after the last sequence."""
    D = S.shape[0]
    p = D - 1
    while p >= 0 and S[p] >= ub[p]:
        p -= 1
    if p < 0:
        return False
    S[p] += 1
    for q in range(p + 1, D):
        S[q] = S[p]
    return True


def enumerate_layer(ub):
    """all valid sequences of a layer in lexicographic order (tests only)."""
    D = len(ub)
    out = []
    for S in itertools.product(*[range(u + 1) for u in ub]):
        if all(S[i] <= S[i + 1] for i in range(D - 1)):
            out.append(S)
    return out
