"""AList 断流重试、流式校验和 latest 提交边界的故障回归。"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import Mock

import pytest
import requests

from dl_helper.training.remote import AListArtifactStore, ArtifactStoreError


class StreamResponse:
    def __init__(self, chunks=(), failure=None, status_code=200):
        self.chunks = chunks
        self.failure = failure
        self.status_code = status_code
        self.closed = False

    @property
    def content(self):
        raise AssertionError("校验不得将整个响应载入内存")

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield from self.chunks
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.closed = True


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("dl_helper.training.remote.time.sleep", Mock())
    result = AListArtifactStore(
        host="https://alist.example.invalid", base_path="/dlh",
        secret_resolver=None, user_secret_key="user", password_secret_key="password",
        connect_timeout=1, read_timeout=1, max_attempts=3, failure_policy="required",
    )
    result._token = "test-token"
    result._session = Mock()
    result._get_info = Mock(return_value={"raw_url": "/d/archive.tar.gz"})
    return result


@pytest.mark.parametrize("read_hash", [False, True])
@pytest.mark.parametrize("failure_type", [
    requests.exceptions.ChunkedEncodingError,
    requests.ConnectionError,
    requests.Timeout,
])
def test_interrupted_body_restarts_and_discards_partial_state(store, read_hash, failure_type):
    partial = StreamResponse([b"discarded-prefix"], failure_type("connection interrupted"))
    complete = StreamResponse([b"actual-", b"payload"])
    store._session.get.side_effect = [partial, complete]
    # 断流后重新获取 raw_url，避免沿用已失效的临时下载地址。
    store._get_info.side_effect = [
        {"raw_url": "/d/old-url"}, {"raw_url": "/d/new-url"},
    ]
    result = (store._raw_read_sha256 if read_hash else store._raw_read)("/archive.tar.gz")
    expected = hashlib.sha256(b"actual-payload").hexdigest() if read_hash else b"actual-payload"
    assert result == expected
    assert store._session.get.call_count == 2
    assert store._get_info.call_count == 2
    assert partial.closed and complete.closed
    assert [call.args[0] for call in store._session.get.call_args_list] == [
        "https://alist.example.invalid/d/old-url",
        "https://alist.example.invalid/d/new-url",
    ]
    assert all(call.kwargs["stream"] for call in store._session.get.call_args_list)


def test_interrupted_body_exhaustion_keeps_original_exception(store):
    responses = [StreamResponse([b"partial"], requests.exceptions.ChunkedEncodingError("broken"))
                 for _ in range(3)]
    store._session.get.side_effect = responses
    with pytest.raises(ArtifactStoreError, match="重试耗尽") as error:
        store._raw_read_sha256("/archive.tar.gz")
    assert isinstance(error.value.__cause__, requests.exceptions.ChunkedEncodingError)
    assert store._session.get.call_count == 3
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_raw_business_error_fails_without_retry(store, status):
    response = StreamResponse(status_code=status)
    store._session.get.return_value = response
    with pytest.raises(ArtifactStoreError, match=str(status)):
        store._raw_read_sha256("/archive.tar.gz")
    assert store._session.get.call_count == 1
    assert response.closed


def test_raw_server_error_retries_and_closes_responses(store):
    failed = StreamResponse(status_code=503)
    succeeded = StreamResponse([b"ok"])
    store._session.get.side_effect = [failed, succeeded]
    assert store._raw_read("/archive.tar.gz") == b"ok"
    assert store._session.get.call_count == 2
    assert failed.closed and succeeded.closed


def test_api_response_body_interruption_retries(store):
    complete = Mock(status_code=200)
    store._session.request.side_effect = [
        requests.exceptions.ChunkedEncodingError("interrupted API body"), complete,
    ]
    assert store._request("PUT", "/api/fs/put", data=b"payload") is complete
    assert store._session.request.call_count == 2


def test_read_retry_does_not_repeat_upload(store):
    store._upload = Mock()
    interrupted = StreamResponse([b"partial"], requests.exceptions.ChunkedEncodingError("broken"))
    complete = StreamResponse([b"payload"])
    store._session.get.side_effect = [interrupted, complete]
    result = store._publish_bytes_with_verify("/remote", b"payload", "archive.tar.gz")
    assert result == hashlib.sha256(b"payload").hexdigest()
    store._upload.assert_called_once_with("/remote/archive.tar.gz", b"payload", 7)
    assert store._session.get.call_count == 2


def test_failed_readback_preserves_previous_remote_latest(store, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint-manifest.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8",
    )
    remote = {"/dlh/runs/run-1/checkpoints/latest.json": b"previous-verified-checkpoint"}
    store._ensure_dir = Mock()
    store._upload = Mock(side_effect=lambda path, data, size: remote.__setitem__(path, data))
    store._session.get.side_effect = [
        StreamResponse([b"partial"], requests.exceptions.ChunkedEncodingError("broken"))
        for _ in range(3)
    ]
    with pytest.raises(ArtifactStoreError, match="重试耗尽"):
        store.publish_checkpoint(str(checkpoint), "run-1", "new-checkpoint")
    assert remote["/dlh/runs/run-1/checkpoints/latest.json"] == b"previous-verified-checkpoint"
    assert store._upload.call_count == 1
    assert store._session.get.call_count == 3


def test_checksum_mismatch_is_not_retried(store):
    store._upload = Mock()
    response = StreamResponse([b"tampered"])
    store._session.get.return_value = response
    with pytest.raises(ArtifactStoreError, match="checksum 不匹配"):
        store._publish_bytes_with_verify("/remote", b"original", "archive.tar.gz")
    assert store._session.get.call_count == 1
    assert response.closed


def test_external_raw_url_does_not_receive_alist_token(store):
    store._get_info.return_value = {"raw_url": "https://storage.example.invalid/archive"}
    store._session.get.return_value = StreamResponse([b"payload"])
    assert store._raw_read("/archive") == b"payload"
    assert store._session.get.call_args.kwargs["headers"] == {}


@pytest.mark.parametrize("retain", [False, True])
def test_zip_bundle_has_one_readback_and_retains_verified_bytes(store, tmp_path, retain):
    run_dir = tmp_path / "run"
    (run_dir / "services").mkdir(parents=True)
    (run_dir / "services" / "service-manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.txt").write_text("训练成果", encoding="utf-8")
    retained = tmp_path / "retained"
    if retain:
        store._retained_bundle_dir = str(retained)
    store._ensure_dir = Mock()
    uploaded = {}

    def upload(path, data, size):
        assert len(data) == size
        uploaded[path] = data
        store._session.get.return_value = StreamResponse([data])

    store._upload = Mock(side_effect=upload)
    result = store.publish_run_bundle(str(run_dir), "run-1")
    payload = uploaded["/dlh/runs/run-1/run-bundle.zip"]
    assert result["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert store._session.get.call_count == 1
    if retain:
        assert (retained / "run-bundle.zip").read_bytes() == payload
        assert list(retained.iterdir()) == [retained / "run-bundle.zip"]
