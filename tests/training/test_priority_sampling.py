"""任务 4.2：64-bit hash 确定性抽样与跨 rank 合并。"""
from __future__ import annotations

import numpy as np

from dl_helper.training.artifacts import _stable_hash, merge_topk_candidates, priority_sample


def test_stable_hash_is_64bit_and_stable():
    h1 = _stable_hash(42, "val", 0)
    h2 = _stable_hash(42, "val", 0)
    assert h1 == h2
    assert 0 <= h1 < 2**64
    assert _stable_hash(42, "val", 0) != _stable_hash(43, "val", 0)
    assert _stable_hash(42, "val", 0) != _stable_hash(42, "test", 0)


def test_sampling_respects_limit():
    n = 1000
    idx = priority_sample(n, 1, "val", 100, sample_ids=None)
    assert len(idx) == 100
    # hash 最小的前 100
    keys = np.array([_stable_hash(1, "val", i) for i in range(n)])
    sorted_keys = np.sort(keys)
    kept = np.array([_stable_hash(1, "val", i) for i in idx])
    assert kept.max() <= sorted_keys[99]


def test_sample_ids_based_sampling():
    ids = np.arange(50, 0, -1)  # 乱序 ID
    idx = priority_sample(50, 5, "val", 10, sample_ids=ids)
    # 用 ID 而非位置
    keys = np.array([_stable_hash(5, "val", sid) for sid in ids])
    assert keys[idx].max() <= np.sort(keys)[9]


def test_merge_topk_more_than_needed():
    per_rank = [np.array([10, 20]), np.array([30, 40]), np.array([50, 60])]
    merged = merge_topk_candidates(per_rank, 2)
    assert len(merged) == 2
