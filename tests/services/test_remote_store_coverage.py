"""补充 remote.py 覆盖率：Local/bundle/fetch/archive 安全/工厂。"""
from __future__ import annotations

import json
import os
import tarfile
import io

import pytest

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.remote import (
    AListArtifactStore,
    ArtifactStoreError,
    AsyncArtifactSync,
    LocalArtifactStore,
    _archive_relative_files,
    _extract_tar_gz_safe,
    _make_tar_gz,
    build_artifact_stores,
)


def test_local_store_methods(tmp_path):
    store = LocalArtifactStore()
    store.publish_checkpoint(str(tmp_path), "r", "c")
    store.publish_run_bundle(str(tmp_path), "r")
    store.publish_sweep_bundle(str(tmp_path), "s")
    assert store.fetch_latest_checkpoint("r", str(tmp_path)) is None
    store.flush()
    assert store.sync_error is None


def test_make_tar_gz_and_extract(tmp_path):
    root = tmp_path / "root"
    os.makedirs(root, exist_ok=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    os.makedirs(root / "sub", exist_ok=True)
    (root / "sub" / "b.txt").write_text("world", encoding="utf-8")
    blob = _make_tar_gz(str(root), exclude_prefixes=())
    target = tmp_path / "out"
    os.makedirs(target, exist_ok=True)
    _extract_tar_gz_safe(blob, str(target))
    assert (target / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (target / "sub" / "b.txt").read_text(encoding="utf-8") == "world"
    assert sorted(_archive_relative_files(str(root))) == ["a.txt", os.path.join("sub", "b.txt")]


def test_make_tar_gz_excludes_prefixes(tmp_path):
    root = tmp_path / "root"
    os.makedirs(root, exist_ok=True)
    (root / "a.txt").write_text("x", encoding="utf-8")
    os.makedirs(root / "checkpoints", exist_ok=True)
    (root / "checkpoints" / "c").write_text("y", encoding="utf-8")
    blob = _make_tar_gz(str(root), exclude_prefixes=("checkpoints",))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = tar.getnames()
    assert "a.txt" in names
    assert not any("checkpoints" in n for n in names)


def test_archive_rejects_symlink(tmp_path):
    root = tmp_path / "root"
    os.makedirs(root, exist_ok=True)
    target = tmp_path / "t"
    target.write_text("x", encoding="utf-8")
    if os.name == "nt":
        pytest.skip("Windows symlink 权限受限")
    os.symlink(str(target), str(root / "link"))
    with pytest.raises(ArtifactStoreError):
        _make_tar_gz(str(root), exclude_prefixes=())


def test_build_artifact_stores_none(tmp_path):
    schema = default_schema()
    cfg = parse_config(schema)
    from dl_helper.training.artifacts import RunLayout
    layout = RunLayout(str(tmp_path / "runs" / "x"))
    layout.ensure()
    stores, async_sync = build_artifact_stores(cfg, None, None, layout)
    assert len(stores) == 1
    assert async_sync is None


class _FakeSession:
    def __init__(self):
        self.remote = {}

    def _handle(self, method, url, **kwargs):
        import urllib.parse
        if "/api/auth/login" in url:
            return _Resp({"code": 200, "data": {"token": "t"}})
        if "/api/fs/mkdir" in url:
            return _Resp({"code": 200})
        if "/api/fs/put" in url:
            path = urllib.parse.unquote(kwargs.get("headers", {}).get("File-Path", ""))
            if not path:
                return _Resp({"code": 500, "message": "missing File-Path", "data": None})
            self.remote[path] = kwargs.get("data", b"")
            return _Resp({"code": 200})
        if "/api/fs/get" in url:
            path = urllib.parse.unquote(url.split("path=", 1)[1].split("&", 1)[0])
            if path not in self.remote:
                return _Resp({"code": 500, "message": "object not found", "data": None})
            content = self.remote[path]
            raw_url = f"/d{urllib.parse.quote(path, safe='/')}"
            return _Resp({"code": 200, "data": {"size": len(content), "raw_url": raw_url}})
        if "/d/" in url:
            path = urllib.parse.unquote(url.split("/d", 1)[1].split("?", 1)[0])
            if path not in self.remote:
                return _Resp({"code": 404}, status_code=404)
            return _Resp(None, content=self.remote[path])
        return _Resp({"code": 404})

    def request(self, method, url, **kwargs):
        return self._handle(method, url, **kwargs)

    def post(self, url, **kwargs):
        return self._handle("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._handle("GET", url, **kwargs)


class _Resp:
    def __init__(self, json_data, content=b"", status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class _Resolver:
    def resolve(self, key):
        return "secret-value"

    def redact(self, t):
        return t


def _alist(tmp_path, session):
    store = AListArtifactStore(
        host="https://alist.example.invalid", base_path="/dlh",
        secret_resolver=_Resolver(), user_secret_key="ALIST_USER",
        password_secret_key="ALIST_PWD", connect_timeout=1, read_timeout=1,
        max_attempts=2, failure_policy="required",
    )
    store._session = session
    return store


def _make_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    os.makedirs(run_dir, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "metrics" / "summary.json").write_text('{"s":1}', encoding="utf-8")
    (run_dir / "run-manifest.json").write_text('{"status":"ok"}', encoding="utf-8")
    os.makedirs(run_dir / "services", exist_ok=True)
    (run_dir / "services" / "service-manifest.json").write_text("{}", encoding="utf-8")
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    (run_dir / "checkpoints" / "c").write_text("x", encoding="utf-8")
    return str(run_dir)


def test_publish_run_bundle(tmp_path):
    session = _FakeSession()
    store = _alist(tmp_path, session)
    run_dir = _make_run_dir(tmp_path)
    store.publish_run_bundle(run_dir, "run-1")
    # bundle 排除了 checkpoints
    assert any("run-bundle.tar.gz" in p for p in session.remote)
    assert not any("checkpoints" in p and "tar" not in p for p in session.remote)


def test_publish_sweep_bundle(tmp_path):
    session = _FakeSession()
    store = _alist(tmp_path, session)
    sweep_dir = tmp_path / "sweep"
    os.makedirs(sweep_dir, exist_ok=True)
    (sweep_dir / "trials.jsonl").write_text("[]", encoding="utf-8")
    os.makedirs(sweep_dir / "services", exist_ok=True)
    (sweep_dir / "services" / "service-manifest.json").write_text("{}", encoding="utf-8")
    store.publish_sweep_bundle(str(sweep_dir), "sweep-1")
    assert any("sweep-bundle.tar.gz" in p for p in session.remote)


def test_fetch_latest_checkpoint(tmp_path):
    session = _FakeSession()
    store = _alist(tmp_path, session)
    # 预置 latest
    ckpt_path = "/dlh/runs/r1/checkpoints/epoch-1/archive.tar.gz"
    session.remote[ckpt_path] = _make_tar_gz(
        str(_make_checkpoint_dir(tmp_path)), exclude_prefixes=())
    session.remote["/dlh/runs/r1/checkpoints/latest.json"] = json.dumps(
        {"path": "epoch-1"}).encode("utf-8")
    target = tmp_path / "staging"
    os.makedirs(target, exist_ok=True)
    result = store.fetch_latest_checkpoint("r1", str(target))
    assert result == "epoch-1"


def _make_checkpoint_dir(tmp_path):
    from dl_helper.training.artifacts import sha256_file

    ckpt = tmp_path / "ckpt"
    os.makedirs(ckpt, exist_ok=True)
    (ckpt / "estimator.joblib").write_bytes(b"model")
    manifest = {
        "complete": True,
        "run_id": "r1",
        "files": {
            "estimator.joblib": {
                "size": os.path.getsize(ckpt / "estimator.joblib"),
                "sha256": sha256_file(str(ckpt / "estimator.joblib")),
            }
        },
    }
    (ckpt / "checkpoint-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return str(ckpt)


def test_async_artifact_sync_pending_merge(tmp_path):
    import threading

    class _SlowStore:
        def __init__(self):
            self.published = []

        def publish_checkpoint(self, local_dir, run_id, ckpt_id):
            self.published.append((local_dir, run_id, ckpt_id))

    store = _SlowStore()
    sync = AsyncArtifactSync(store)
    sync.submit_checkpoint("d0", "r", "c0")
    sync.submit_checkpoint("d1", "r", "c1")
    sync.flush(terminal=True)
    assert len(store.published) >= 1
