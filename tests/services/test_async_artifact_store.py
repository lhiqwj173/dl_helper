"""任务 5.3：有界异步 AList 同步 —— pending 合并、active 不取消、异常传播、join。"""
from __future__ import annotations

import threading
import time

import pytest

from dl_helper.training.remote import AsyncArtifactSync


class _SlowStore:
    """记录发布；可阻塞/报错。"""

    def __init__(self):
        self.published: list[tuple[str, str, str]] = []
        self.error: Exception | None = None
        self.lock = threading.Lock()

    def publish_checkpoint(self, local_dir, run_id, ckpt_id):
        with self.lock:
            if self.error:
                raise self.error
            time.sleep(0.05)
            self.published.append((local_dir, run_id, ckpt_id))


def test_pending_merge_capacity_one():
    store = _SlowStore()
    sync = AsyncArtifactSync(store)
    sync.submit_checkpoint("d0", "r", "c0")
    sync.submit_checkpoint("d1", "r", "c1")  # c0 active/pending → c1 合并
    sync.submit_checkpoint("d2", "r", "c2")  # 容量 1，替换 pending
    sync.flush(terminal=True)
    ids = [c[2] for c in store.published]
    # c2 保留（最新），c1 可能被合并替换；全部本地保留
    assert "c2" in ids


def test_async_error_propagates_on_flush():
    store = _SlowStore()
    store.error = RuntimeError("upload failed")
    sync = AsyncArtifactSync(store)
    sync.submit_checkpoint("d", "r", "c")
    with pytest.raises(RuntimeError):
        sync.flush(terminal=True)


def test_join_leaves_no_residual_threads():
    store = _SlowStore()
    sync = AsyncArtifactSync(store)
    sync.submit_checkpoint("d", "r", "c0")
    sync.flush(terminal=True)
    threads = [t for t in threading.enumerate() if t is not threading.main_thread()
               and "worker" in t.name or t.daemon is False]
    # 非 daemon 后台线程应已 join
    active = [t for t in threading.enumerate() if t is not threading.main_thread()]
    assert all(t.daemon for t in active), f"存在非 daemon 残留线程: {active}"


def test_terminal_waits_for_all():
    store = _SlowStore()
    sync = AsyncArtifactSync(store)
    sync.submit_checkpoint("d", "r", "c0")
    sync.flush(terminal=True)
    assert len(store.published) == 1
