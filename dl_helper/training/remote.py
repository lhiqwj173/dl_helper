"""ArtifactStore：Local 始终启用；AList 发布 checkpoint/run/sweep；有界异步同步。"""
from __future__ import annotations

import io
import gzip
import hashlib
import json
import os
import queue
import tarfile
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .artifacts import RunLayout, sha256_file
from .config import Config

RETRY_BACKOFF = (2.0, 4.0, 8.0)


class ArtifactStoreError(Exception):
    """远程 Artifact 发布/获取失败。"""


def _archive_relative_files(root: str) -> list[str]:
    """归档 run/sweep 根内 regular file；拒绝 symlink/absolute/..。"""
    from .artifacts import list_relative_files

    rels = list_relative_files(root)
    out = []
    for rel in rels:
        if rel.startswith("."):
            continue  # 排除 staging/tmp/lock
        if rel.replace("\\", "/") == "services/bundle-published.json":
            continue  # 运行态 marker 不属于 immutable bundle
        if "\\" in rel or "/" in rel:
            parts = rel.replace("\\", "/").split("/")
            if any(p == ".." for p in parts):
                raise ArtifactStoreError(f"归档成员含 .. : {rel!r}")
        out.append(rel)
    return out


def bundle_checksum_for_directory(root: str, exclude_prefixes: tuple[str, ...] = ()) -> str:
    """按归档文件内容计算稳定 bundle checksum，不受 tar/gzip 时间元数据影响。"""
    files: dict[str, dict[str, Any]] = {}
    for rel in _archive_relative_files(root):
        normalized = rel.replace("\\", "/")
        if any(normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/")
               for prefix in exclude_prefixes):
            continue
        path = os.path.join(root, rel)
        files[normalized] = {"size": os.path.getsize(path), "sha256": sha256_file(path)}
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_tar_gz(root: str, exclude_prefixes: tuple[str, ...]) -> bytes:
    """构建元数据固定的 tar.gz；相同文件内容产生相同 archive checksum。"""
    tar_buf = io.BytesIO()
    rels = _archive_relative_files(root)
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        for rel in rels:
            if any(rel == prefix or rel.startswith(prefix + "/") or rel.startswith(prefix + "\\")
                   for prefix in exclude_prefixes):
                continue
            full = os.path.join(root, rel)
            if os.path.islink(full):
                raise ArtifactStoreError(f"归档成员为符号链接: {rel!r}")
            info = tar.gettarinfo(full, arcname=rel)
            if not info.isfile():
                raise ArtifactStoreError(f"归档成员不是 regular file: {rel!r}")
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with open(full, "rb") as f:
                tar.addfile(info, f)
    return gzip.compress(tar_buf.getvalue(), compresslevel=1, mtime=0)


class LocalArtifactStore:
    """本地存储始终启用；文件已就地写入，publish 为验证性 no-op。"""

    def __init__(self) -> None:
        self.sync_error: Exception | None = None

    def publish_checkpoint(self, local_dir: str, run_id: str, checkpoint_id: str) -> None:
        return None

    def publish_run_bundle(self, local_dir: str, run_id: str) -> dict[str, str]:
        return None

    def publish_sweep_bundle(self, local_dir: str, sweep_id: str) -> None:
        return None

    def fetch_latest_checkpoint(self, run_id: str, target_dir: str) -> str | None:
        return None

    def flush(self) -> None:
        return None


# --------------------------------------------------------------------------
# AListArtifactStore
# --------------------------------------------------------------------------

class AListArtifactStore:
    """AList 客户端：显式 host/base_path、Secret、timeout、有限重试、回读 SHA。"""

    def __init__(
        self,
        host: str,
        base_path: str,
        secret_resolver: Any,
        user_secret_key: str,
        password_secret_key: str,
        connect_timeout: float,
        read_timeout: float,
        max_attempts: int,
        failure_policy: str,
    ) -> None:
        parsed_host = urlparse(host)
        if parsed_host.scheme not in ("http", "https") or not parsed_host.netloc:
            raise ArtifactStoreError(
                f"AList host 必须是带主机的 HTTP(S) URL: {host!r}"
            )
        self._host = host.rstrip("/")
        self._base_path = base_path.rstrip("/")
        self._resolver = secret_resolver
        self._user_key = user_secret_key
        self._password_key = password_secret_key
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_attempts = max_attempts
        self._policy = failure_policy
        self._session = None
        self._token: str | None = None
        self.sync_error: Exception | None = None

    def secret_keys(self) -> list[str]:
        """OSR-002：AList 使用的 Secret key 名（供启用服务前的预检解析）。"""
        return [self._user_key, self._password_key]

    def _get_session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _login(self) -> None:
        import requests

        session = self._get_session()
        username = self._resolver.resolve(self._user_key)
        password = self._resolver.resolve(self._password_key)
        resp = session.post(
            f"{self._host}/api/auth/login",
            json={"username": username, "password": password},
            timeout=(self._connect_timeout, self._read_timeout),
        )
        if resp.status_code in (401, 403):
            raise ArtifactStoreError(f"AList 认证失败: HTTP {resp.status_code}（不重试）")
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") not in (200, None) or not data.get("data", {}).get("token"):
            raise ArtifactStoreError(f"AList 登录业务码非法: {data.get('code')!r}")
        self._token = data["data"]["token"]

    def _request(self, method: str, path: str, *, retryable: bool = True, **kwargs) -> Any:
        """有限重试：401/403/业务参数错误不重试；连接/5xx 按 2/4/8。"""
        import requests

        if self._token is None:
            self._login()
        session = self._get_session()
        request_headers = kwargs.pop("headers", {})
        if not isinstance(request_headers, dict):
            raise ArtifactStoreError("AList 请求 headers 必须是 dict")
        headers = {"Authorization": self._token, **request_headers}
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                resp = session.request(
                    method, f"{self._host}{path}", headers=headers,
                    timeout=(self._connect_timeout, self._read_timeout), **kwargs,
                )
                if resp.status_code in (401, 403):
                    # 认证失效：刷新一次后重试
                    if attempt == 0:
                        self._token = None
                        self._login()
                        headers["Authorization"] = self._token
                        continue
                    raise ArtifactStoreError(f"AList 认证失败: HTTP {resp.status_code}")
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"AList HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise ArtifactStoreError(f"AList 业务/参数错误: HTTP {resp.status_code}（不重试）")
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if not retryable or attempt >= self._max_attempts - 1:
                    break
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
        raise ArtifactStoreError(f"AList 请求失败（重试耗尽）: {path}") from last_exc

    def _ensure_dir(self, remote_path: str) -> None:
        resp = self._request("POST", "/api/fs/mkdir", json={"path": remote_path},
                             retryable=False)
        data = resp.json()
        code = data.get("code")
        if code != 200:
            raise ArtifactStoreError(
                f"AList mkdir 失败: code={code!r}, message={data.get('message')!r}"
            )

    def _get_info(self, remote_path: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        resp = self._request("GET", f"/api/fs/get?path={_quote(remote_path)}", retryable=False)
        body = resp.json()
        code = body.get("code")
        info = body.get("data")
        if missing_ok and code != 200 and info is None:
            return None
        if code != 200 or not isinstance(info, dict):
            raise ArtifactStoreError(
                f"AList get 失败: code={code!r}, message={body.get('message')!r}, "
                f"path={remote_path!r}"
            )
        return info

    def _upload(self, remote_path: str, data: bytes, size: int) -> None:
        resp = self._request(
            "PUT",
            "/api/fs/put",
            headers={"File-Path": _quote(remote_path), "Content-Type": "application/octet-stream"},
            data=data,
            retryable=True,
        )
        body = resp.json()
        if body.get("code") != 200:
            raise ArtifactStoreError(
                f"AList put 失败: code={body.get('code')!r}, message={body.get('message')!r}"
            )
        # 轮询 info 到 size 匹配
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            info = self._get_info(remote_path, missing_ok=True)
            if info is None:
                time.sleep(0.5)
                continue
            if info.get("size") == size:
                return
            time.sleep(0.5)
        raise ArtifactStoreError(f"AList 上传后 size 未匹配: {remote_path}")

    def _raw_read(self, remote_path: str) -> bytes:
        import requests
        from urllib.parse import urljoin

        info = self._get_info(remote_path)
        assert info is not None
        raw_url = info.get("raw_url")
        if not isinstance(raw_url, str) or not raw_url:
            raise ArtifactStoreError(f"AList metadata 缺少 raw_url: {remote_path!r}")
        url = urljoin(f"{self._host}/", raw_url)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ArtifactStoreError(f"AList raw_url 必须是带主机的 HTTP(S) URL: {url!r}")

        host = urlparse(self._host)
        headers = {"Authorization": self._token} if parsed.netloc == host.netloc else {}
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                resp = self._get_session().get(
                    url,
                    headers=headers,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
                if resp.status_code in (401, 403):
                    raise ArtifactStoreError(f"AList raw 下载认证失败: HTTP {resp.status_code}")
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"AList raw HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise ArtifactStoreError(f"AList raw 下载失败: HTTP {resp.status_code}")
                return resp.content
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if attempt >= self._max_attempts - 1:
                    break
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
        raise ArtifactStoreError(f"AList raw 下载失败（重试耗尽）: {remote_path}") from last_exc

    def _raw_read_sha256(self, remote_path: str) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(self._raw_read(remote_path))
        return h.hexdigest()

    def _publish_file_with_verify(self, remote_dir: str, local_path: str, rel: str) -> None:
        """上传 → size 匹配 → raw 回读 SHA 校验。"""
        remote_path = f"{remote_dir}/{rel}"
        local_size = os.path.getsize(local_path)
        with open(local_path, "rb") as f:
            data = f.read()
        self._upload(remote_path, data, local_size)
        remote_sha = self._raw_read_sha256(remote_path)
        if remote_sha != sha256_file(local_path):
            raise ArtifactStoreError(f"AList 回读 checksum 不匹配: {rel}")

    def _publish_tar_gz(self, remote_dir: str, local_root: str, name: str,
                        exclude_prefixes: tuple[str, ...]) -> dict[str, str]:
        blob = _make_tar_gz(local_root, exclude_prefixes)
        bundle_checksum = bundle_checksum_for_directory(local_root, exclude_prefixes)
        self._ensure_dir(remote_dir)
        self._publish_file_with_verify(remote_dir, _bytes_to_temp(blob), f"{name}.tar.gz")
        # 回读校验 archive SHA
        remote_sha = self._raw_read_sha256(f"{remote_dir}/{name}.tar.gz")
        import hashlib
        local_sha = hashlib.sha256(blob).hexdigest()
        if remote_sha != local_sha:
            raise ArtifactStoreError("AList archive 回读 SHA 不匹配")
        return {"bundle_checksum": bundle_checksum, "archive_sha256": local_sha}

    def publish_checkpoint(self, local_dir: str, run_id: str, checkpoint_id: str) -> None:
        remote_dir = f"{self._base_path}/runs/{run_id}/checkpoints/{checkpoint_id}"
        self._publish_tar_gz(remote_dir, local_dir, "archive", exclude_prefixes=())
        # manifest 回读校验
        self._publish_file_with_verify(remote_dir, os.path.join(local_dir, "checkpoint-manifest.json"),
                                       "checkpoint-manifest.json")
        # latest 最后发布
        latest = {"schema_version": 1, "checkpoint_id": checkpoint_id, "path": checkpoint_id}
        self._publish_file_with_verify(f"{self._base_path}/runs/{run_id}/checkpoints",
                                       _bytes_to_temp(json.dumps(latest, ensure_ascii=False).encode("utf-8")),
                                       "latest.json")

    def fetch_latest_checkpoint(self, run_id: str, target_dir: str) -> str | None:
        from .artifacts import move_tree, read_json, remove_tree
        from .checkpoint import CHECKPOINT_MANIFEST, update_latest, validate_manifest_complete

        latest_path = f"{self._base_path}/runs/{run_id}/checkpoints/latest.json"
        info = self._get_info(latest_path, missing_ok=True)
        if info is None:
            return None
        try:
            latest = json.loads(self._raw_read(latest_path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError("AList latest.json 不是合法 UTF-8 JSON") from exc
        if not isinstance(latest, dict):
            raise ArtifactStoreError("AList latest.json 根必须是对象")
        ckpt_path = latest.get("path")
        if not isinstance(ckpt_path, str) or not ckpt_path:
            raise ArtifactStoreError("AList latest.json 缺少非空 path")
        if ckpt_path in (".", "..") or os.path.basename(ckpt_path) != ckpt_path:
            raise ArtifactStoreError(f"AList latest.json path 不是单个安全路径段: {ckpt_path!r}")
        # 下载 archive 到隔离 staging
        archive_path = f"{self._base_path}/runs/{run_id}/checkpoints/{ckpt_path}/archive.tar.gz"
        os.makedirs(target_dir, exist_ok=True)
        staging_root = os.path.join(target_dir, f".remote-staging-{os.getpid()}")
        staging_checkpoint = os.path.join(staging_root, ckpt_path)
        destination = os.path.join(target_dir, ckpt_path)
        if os.path.exists(staging_root):
            raise ArtifactStoreError(f"AList 恢复 staging 已存在: {staging_root!r}")
        try:
            os.makedirs(staging_checkpoint)
            _extract_tar_gz_safe(self._raw_read(archive_path), staging_checkpoint)
            manifest_path = os.path.join(staging_checkpoint, CHECKPOINT_MANIFEST)
            if not os.path.isfile(manifest_path):
                raise ArtifactStoreError("AList checkpoint archive 缺少 checkpoint-manifest.json")
            manifest = read_json(manifest_path)
            validate_manifest_complete(manifest, staging_checkpoint)
            if manifest.get("run_id") != run_id:
                raise ArtifactStoreError("AList checkpoint manifest run_id 不匹配")
            move_tree(staging_checkpoint, destination)
            update_latest(target_dir, ckpt_path, str(latest.get("checkpoint_id", ckpt_path)))
            return ckpt_path
        finally:
            remove_tree(staging_root)

    def publish_run_bundle(self, local_dir: str, run_id: str) -> dict[str, str]:
        service_manifest = os.path.join(local_dir, "services", "service-manifest.json")
        if not os.path.isfile(service_manifest):
            raise ArtifactStoreError("发布 run bundle 前缺少最终 service-manifest.json")
        remote_dir = f"{self._base_path}/runs/{run_id}"
        result = self._publish_tar_gz(remote_dir, local_dir, "run-bundle",
                                      exclude_prefixes=("checkpoints",))
        result["service_manifest_sha256"] = sha256_file(service_manifest)
        return result

    def publish_sweep_bundle(self, local_dir: str, sweep_id: str) -> dict[str, str]:
        service_manifest = os.path.join(local_dir, "services", "service-manifest.json")
        if not os.path.isfile(service_manifest):
            raise ArtifactStoreError("发布 sweep bundle 前缺少最终 service-manifest.json")
        remote_dir = f"{self._base_path}/sweeps/{sweep_id}"
        result = self._publish_tar_gz(remote_dir, local_dir, "sweep-bundle", exclude_prefixes=("checkpoints",))
        result["service_manifest_sha256"] = sha256_file(service_manifest)
        return result

    def flush(self) -> None:
        if self.sync_error:
            raise self.sync_error


def _quote(path: str) -> str:
    import urllib.parse
    return urllib.parse.quote(path, safe="/")


def _bytes_to_temp(data: bytes) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(prefix="dlh-upload-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _extract_tar_gz_safe(blob: bytes, target_dir: str) -> None:
    """安全解压：拒绝 symlink、绝对路径、..。"""
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ArtifactStoreError(f"远程 archive 含链接: {member.name!r}")
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ArtifactStoreError(f"远程 archive 含危险成员: {member.name!r}")
            member.name = os.path.basename(member.name) if "/" not in member.name else member.name.lstrip("/")
        tar.extractall(target_dir, filter="data")


# --------------------------------------------------------------------------
# 有界异步同步（任务 5.3）
# --------------------------------------------------------------------------

class AsyncArtifactSync:
    """主进程唯一非 daemon worker；容量 1 pending checkpoint；terminal 等待全部。"""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._queue: queue.Queue[tuple[str, str, str] | None] = queue.Queue(maxsize=1)
        self._active: threading.Thread | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            local_dir, run_id, ckpt_id = item
            try:
                self._store.publish_checkpoint(local_dir, run_id, ckpt_id)
            except Exception as exc:  # 存入同步器，下一安全边界抛出
                with self._lock:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _ensure_worker(self) -> None:
        if self._active is None or not self._active.is_alive():
            self._active = threading.Thread(target=self._worker, daemon=False)
            self._active.start()

    def submit_checkpoint(self, local_dir: str, run_id: str, checkpoint_id: str) -> None:
        """提交 checkpoint；capacity 1，新项替换尚未开始的旧 pending，不取消 active。"""
        self._ensure_worker()
        item = (local_dir, run_id, checkpoint_id)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # 替换尚未开始的 pending 项（队列只存 pending，active 已被 worker get 移除）
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def _check_error(self) -> None:
        with self._lock:
            if self._error is not None:
                exc = self._error
                raise exc

    def flush(self, terminal: bool = False) -> None:
        """等待当前队列处理；terminal 时先完全 join 再检查错误。"""
        self._queue.join()
        if terminal:
            self._stop_worker()
        self._check_error()

    def join(self) -> None:
        self._stop_worker()

    def _stop_worker(self) -> None:
        if self._active is None:
            return
        self._queue.put(None)
        self._active.join()
        self._active = None
        self._check_error()


def build_artifact_stores(config: Config, platform: Any, secret_resolver: Any, layout: RunLayout):
    """构建 store 列表（Local 始终启用；alist 时附加 AListArtifactStore + 异步）。"""
    stores = [LocalArtifactStore()]
    async_sync = None
    if config.remote.type == "alist":
        remote = config.remote
        alist = AListArtifactStore(
            host=remote.host,
            base_path=remote.base_path,
            secret_resolver=secret_resolver,
            user_secret_key=remote.user_secret_key,
            password_secret_key=remote.password_secret_key,
            connect_timeout=remote.connect_timeout_seconds,
            read_timeout=remote.read_timeout_seconds,
            max_attempts=remote.max_attempts,
            failure_policy=remote.failure_policy,
        )
        stores.append(alist)
        if remote.async_upload:
            async_sync = AsyncArtifactSync(alist)
    return stores, async_sync
