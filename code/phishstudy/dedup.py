"""Exact, template and near-duplicate grouping across all collections.

Messages are linked when their canonical text is identical, when they are
identical after digits are masked (template match), or when the Jaccard
similarity of their word 5-shingle sets is at least ``threshold``. Candidate
near-duplicate pairs come from MinHash banding and are confirmed with the exact
Jaccard similarity of the shingle sets. Linked messages form one group
(connected components), which is the unit used for splitting.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def shingles(text: str, k: int = 5) -> np.ndarray:
    tokens = text.split()
    if len(tokens) < k:
        grams = [" ".join(tokens)] if tokens else []
    else:
        grams = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    return np.unique(np.fromiter((zlib.crc32(g.encode("utf-8", "replace")) for g in grams),
                                 dtype=np.uint32, count=len(grams)))


def _minhash_fast(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash with 32-bit arithmetic: h(x) = (a*x + b) mod 2^32 after mixing, a odd."""
    if len(sh) == 0:
        return np.full(len(a), np.iinfo(np.uint32).max, dtype=np.uint32)
    x = sh
    best = np.full(len(a), np.iinfo(np.uint32).max, dtype=np.uint32)
    for start in range(0, len(x), 8192):
        block = x[None, start:start + 8192]
        h = (a[:, None] * block + b[:, None])  # wraps modulo 2^32
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x2C1B3C6D)
        h ^= h >> np.uint32(12)
        best = np.minimum(best, h.min(axis=1))
    return best


def group_messages(canon: list[str], template: list[str], threshold: float = 0.8,
                   num_perm: int = 128, bands: int = 16, seed: int = 20261105):
    """Return (group index per message, list of confirmed near-duplicate pairs with similarity)."""
    n = len(canon)
    uf = UnionFind(n)
    for keys in (canon, template):
        first = {}
        for i, key in enumerate(keys):
            j = first.setdefault(key, i)
            if j != i:
                uf.union(i, j)
    rng = np.random.default_rng(seed)
    a = (rng.integers(1, 2**32, size=num_perm, dtype=np.uint64) | 1).astype(np.uint32)
    b = rng.integers(0, 2**32, size=num_perm, dtype=np.uint64).astype(np.uint32)
    sets = [shingles(t) for t in template]
    sigs = np.vstack([_minhash_fast(s, a, b) for s in sets]) if n else np.empty((0, num_perm), np.uint32)
    rows = num_perm // bands
    candidates = set()
    for band in range(bands):
        buckets = defaultdict(list)
        block = sigs[:, band * rows:(band + 1) * rows]
        for i in range(n):
            if len(sets[i]) == 0:
                continue
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) > 1:
                anchor = members[0]
                for j in members[1:]:
                    candidates.add((anchor, j) if anchor < j else (j, anchor))
                # pairs among non-anchor members are reached through later bands or transitivity
                if len(members) <= 50:
                    for x in range(1, len(members)):
                        for y in range(x + 1, len(members)):
                            candidates.add((members[x], members[y]))
    pairs = []
    for i, j in candidates:
        si, sj = sets[i], sets[j]
        inter = len(np.intersect1d(si, sj, assume_unique=True))
        union = len(si) + len(sj) - inter
        sim = inter / union if union else 0.0
        if sim >= threshold:
            pairs.append((i, j, sim))
            uf.union(i, j)
    groups = np.array([uf.find(i) for i in range(n)])
    return groups, pairs
