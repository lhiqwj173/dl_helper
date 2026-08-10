"""任务 5.2：AListArtifactStore 发布顺序、认证、重试与 checksum。"""
from __future__ import annotations

import io
import json
import os
import tarfile

import pytest

from dl_helper.training.remote import AListArtifactStore, ArtifactStoreError, _extract_tar_gz_safe


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json = json_data
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """模拟 AList API；记录调用序列。"""

    def __init__(self):
        self.calls: list[str] = []
        self.remote: dict[str, bytes] = {}
        self.fail_put = 0  # 前 N 次 put 返回 5xx
        self.put_count = 0
        self.login_count = 0
        self.token = "${ALIST_TOKEN}"
        self.upload_paths: list[str] = []

    def _handle(self, method, url, **kwargs):
        self.calls.append(f"{method} {url}")
        if "/api/auth/login" in url:
            self.login_count += 1
            return _FakeResponse(200, {"code": 200, "data": {"token": self.token}})
        if "/api/fs/mkdir" in url:
            return _FakeResponse(200, {"code": 200})
        if "/api/fs/put" in url:
            self.put_count += 1
            if self.fail_put > 0:
                self.fail_put -= 1
                return _FakeResponse(500)
            import urllib.parse
            path = kwargs.get("headers", {}).get("File-Path")
            if not path:
                return _FakeResponse(200, {"code": 500, "message": "missing File-Path", "data": None})
            decoded_path = urllib.parse.unquote(path)
            self.upload_paths.append(decoded_path)
            self.remote[decoded_path] = kwargs.get("data", b"")
            return _FakeResponse(200, {"code": 200})
        if "/api/fs/get" in url:
            path = url.split("path=", 1)[1].split("&", 1)[0]
            import urllib.parse
            path = urllib.parse.unquote(path)
            if path not in self.remote:
                return _FakeResponse(200, {"code": 500, "message": "object not found", "data": None})
            content = self.remote[path]
            raw_url = f"/d{urllib.parse.quote(path, safe='/')}"
            return _FakeResponse(200, {"code": 200, "data": {"size": len(content), "raw_url": raw_url}})
        if "/d/" in url:
            import urllib.parse
            path = urllib.parse.unquote(url.split("/d", 1)[1].split("?", 1)[0])
            if path not in self.remote:
                return _FakeResponse(404, {"code": 404})
            return _FakeResponse(200, content=self.remote[path])
        return _FakeResponse(404)

    def request(self, method, url, **kwargs):
        return self._handle(method, url, **kwargs)

    def post(self, url, **kwargs):
        return self._handle("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._handle("GET", url, **kwargs)


class _Resolver:
    def __init__(self, env=None):
        self._env = env or {"ALIST_USER": "u", "ALIST_PWD": "p"}

    def resolve(self, key):
        return self._env[key]


def _store(session, **kw):
    store = AListArtifactStore(
        host="https://alist.example.invalid",
        base_path="/dlh",
        secret_resolver=_Resolver(),
        user_secret_key="ALIST_USER",
        password_secret_key="ALIST_PWD",
        connect_timeout=1.0,
        read_timeout=1.0,
        max_attempts=3,
        failure_policy="required",
    )
    store._session = session
    return store


def test_publish_checkpoint_order(tmp_path):
    session = _FakeSession()
    store = _store(session)
    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "estimator.joblib").write_bytes(b"model")
    (ckpt_dir / "checkpoint-manifest.json").write_text('{"complete": true}', encoding="utf-8")
    store.publish_checkpoint(str(ckpt_dir), "run-1", "ck-1")
    # 发布顺序：archive → manifest → latest
    assert session.upload_paths == [
        "/dlh/runs/run-1/checkpoints/ck-1/archive.tar.gz",
        "/dlh/runs/run-1/checkpoints/ck-1/checkpoint-manifest.json",
        "/dlh/runs/run-1/checkpoints/latest.json",
    ]


def test_publish_checkpoint_checksum_mismatch(tmp_path):
    session = _FakeSession()

    class BadResolver(_Resolver):
        pass

    store = AListArtifactStore(
        host="https://alist.example.invalid", base_path="/dlh",
        secret_resolver=_Resolver(), user_secret_key="ALIST_USER",
        password_secret_key="ALIST_PWD", connect_timeout=1.0, read_timeout=1.0,
        max_attempts=2, failure_policy="required",
    )
    store._session = session

    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "estimator.joblib").write_bytes(b"model")
    (ckpt_dir / "checkpoint-manifest.json").write_text('{"complete": true}', encoding="utf-8")
    # 篡改回读内容 → checksum 不匹配
    orig = session._handle

    def tampered(method, url, **kwargs):
        resp = orig(method, url, **kwargs)
        if "/d/" in url and "archive.tar.gz" in url:
            resp.content = b"tampered-bytes"
        return resp

    session._handle = tampered
    with pytest.raises(ArtifactStoreError):
        store.publish_checkpoint(str(ckpt_dir), "run-1", "ck-1")


def test_put_business_error_fails_fast(tmp_path):
    session = _FakeSession()
    original = session._handle

    def put_business_error(method, url, **kwargs):
        if "/api/fs/put" in url:
            return _FakeResponse(200, {"code": 500, "message": "upload rejected", "data": None})
        return original(method, url, **kwargs)

    session._handle = put_business_error
    store = _store(session)
    path = tmp_path / "x.bin"
    path.write_bytes(b"x")
    with pytest.raises(ArtifactStoreError, match="upload rejected"):
        store._publish_file_with_verify("/dlh/probe", str(path), "x.bin")


def test_fetch_latest_checkpoint_uses_raw_url(tmp_path):
    session = _FakeSession()
    store = _store(session)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint-manifest.json").write_text(
        '{"complete": true, "run_id": "run-raw", "files": {}}', encoding="utf-8"
    )
    store.publish_checkpoint(str(checkpoint_dir), "run-raw", "ck-1")

    target = tmp_path / "target"
    checkpoint_id = store.fetch_latest_checkpoint("run-raw", str(target))

    assert checkpoint_id == "ck-1"
    assert (target / "ck-1" / "checkpoint-manifest.json").is_file()
    assert json.loads((target / "latest.json").read_text(encoding="utf-8"))["path"] == "ck-1"
    assert not any(path.name.startswith(".remote-staging-") for path in target.iterdir())


def test_authentication_not_retried(tmp_path):
    session = _FakeSession()

    def auth_fail(method, url, **kwargs):
        if "/api/auth/login" in url:
            return _FakeResponse(401)
        return session._handle(method, url, **kwargs)

    session._handle = auth_fail
    store = _store(session)
    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "a").write_bytes(b"x")
    with pytest.raises(ArtifactStoreError):
        store.publish_checkpoint(str(ckpt_dir), "r", "c")


def test_5xx_retried_then_succeeds(tmp_path):
    session = _FakeSession()
    session.fail_put = 1  # 第一次 put 500
    store = _store(session)
    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "a").write_bytes(b"x")
    (ckpt_dir / "checkpoint-manifest.json").write_text("{}", encoding="utf-8")
    # 重试后成功
    store.publish_checkpoint(str(ckpt_dir), "run-1", "ck-1")
    assert session.put_count >= 2


def test_5xx_exhausts_retries(tmp_path):
    session = _FakeSession()
    session.fail_put = 999  # 一直 500
    store = _store(session)
    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "a").write_bytes(b"x")
    with pytest.raises(ArtifactStoreError):
        store.publish_checkpoint(str(ckpt_dir), "r", "c")


def test_dangerous_archive_rejected():
    """archive 含 symlink/绝对路径/.. 拒绝。"""
    import io as _io
    import tarfile

    buf = _io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../../evil.txt")
        content = b"x"
        info.size = len(content)
        tar.addfile(info, _io.BytesIO(content))
    blob = buf.getvalue()
    with pytest.raises(ArtifactStoreError):
        _extract_tar_gz_safe(blob, "target")


def test_https_host_required():
    with pytest.raises(ArtifactStoreError):
        AListArtifactStore(host="http://insecure.example", base_path="/x",
                           secret_resolver=None, user_secret_key="a", password_secret_key="b",
                           connect_timeout=1, read_timeout=1, max_attempts=2, failure_policy="required")
