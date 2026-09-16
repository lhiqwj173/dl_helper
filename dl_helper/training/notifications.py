"""固定企业微信应用消息客户端：官方 gettoken/message API、token 缓存、2048 UTF-8 上限。"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Mapping

import requests

WECOM_HOST = "https://qyapi.weixin.qq.com"
TOKEN_PATH = "/cgi-bin/gettoken"
SEND_PATH = "/cgi-bin/message/send"
CONTENT_MAX_BYTES = 2048
TOKEN_REFRESH_MARGIN_SECONDS = 120

# 已知 token 失效/过期业务码：只清缓存刷新重放一次
TOKEN_INVALID_CODES = {40014, 42001}

# 业务码：系统繁忙等可重试；其他业务错误不重试
RETRYABLE_CODES = {-1, 45009, 45010}


class WecomError(Exception):
    """企业微信投递失败。"""


class WecomClient:
    """独立企业微信客户端；不导入 py-ext，host 固定，token 不落盘，无全局单例。"""

    def __init__(
        self,
        corp_id_secret_key: str,
        corp_secret_key: str,
        agent_id_secret_key: str,
        to_user: str,
        secret_resolver: Any,
        connect_timeout: float,
        read_timeout: float,
        max_attempts: int,
    ) -> None:
        if not to_user:
            raise WecomError("to_user 必须为非空字符串")
        self._corp_id_key = corp_id_secret_key
        self._corp_secret_key = corp_secret_key
        self._agent_id_key = agent_id_secret_key
        self._to_user = to_user
        self._resolver = secret_resolver
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_attempts = max_attempts
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._lock = threading.Lock()

    def secret_keys(self) -> list[str]:
        """OSR-002：WeCom 使用的 Secret key 名（供启用服务前预检解析）。"""
        return [self._corp_id_key, self._corp_secret_key, self._agent_id_key]

    def _agent_id(self) -> int:
        raw = self._resolver.resolve(self._agent_id_key)
        if not re.match(r"^[1-9]\d*$", raw.strip()):
            raise WecomError(f"agent_id 必须为正十进制整数")
        return int(raw.strip())

    def get_token(self, force: bool = False) -> str:
        with self._lock:
            now = time.monotonic()
            if (not force and self._token is not None and now < self._token_expires_at):
                return self._token
            import requests

            corp_id = self._resolver.resolve(self._corp_id_key)
            corp_secret = self._resolver.resolve(self._corp_secret_key)
            resp = requests.get(
                f"{WECOM_HOST}{TOKEN_PATH}",
                params={"corpid": corp_id, "corpsecret": corp_secret},
                timeout=(self._connect_timeout, self._read_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode") != 0 or not data.get("access_token"):
                raise WecomError(f"gettoken 失败: errcode={data.get('errcode')!r}")
            expires_in = int(data.get("expires_in", 0))
            if expires_in <= 0:
                raise WecomError("gettoken 返回非法 expires_in")
            # 提前量：剩余不足 120s 按 80% 生命周期
            margin = min(TOKEN_REFRESH_MARGIN_SECONDS, expires_in * 0.2)
            self._token = data["access_token"]
            self._token_expires_at = now + expires_in - margin
            return self._token

    def send_text(self, content: str, *, redactor=None) -> dict[str, Any]:
        """发送 UTF-8 text 应用消息；只 errcode=0 成功。"""
        if redactor is not None:
            content = redactor(content)
        content = self._trim_utf8(content)
        payload = {
            "touser": self._to_user,
            "agentid": self._agent_id(),
            "msgtype": "text",
            "text": {"content": content},
        }
        replay_token = False
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                token = self.get_token(force=replay_token)
                data = self._post_send(token, payload)
                code = data.get("errcode")
                if code == 0:
                    return data
                if code in TOKEN_INVALID_CODES and not replay_token:
                    replay_token = True  # 清缓存刷新重放一次
                    continue
                if code in RETRYABLE_CODES and attempt < self._max_attempts - 1:
                    time.sleep(_backoff(attempt))
                    continue
                raise WecomError(f"企业微信 send 业务错误: errcode={code!r}（不重试）")
            except (WecomError, requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if isinstance(exc, WecomError):
                    raise
                last_exc = exc
                if attempt >= self._max_attempts - 1:
                    break
                time.sleep(_backoff(attempt))
        raise WecomError("企业微信 send 失败（重试耗尽）") from last_exc

    def _post_send(self, token: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        import requests

        resp = requests.post(
            f"{WECOM_HOST}{SEND_PATH}",
            params={"access_token": token},
            json=payload,
            timeout=(self._connect_timeout, self._read_timeout),
        )
        resp.raise_for_status()
        return resp.json()

    def _trim_utf8(self, content: str) -> str:
        """按 code point 边界裁剪到 2048 UTF-8 bytes；不切多字节字符。"""
        while len(content.encode("utf-8")) > CONTENT_MAX_BYTES and content:
            content = content[:-1]
        if len(content.encode("utf-8")) > CONTENT_MAX_BYTES:
            raise WecomError("内容仍超过 2048 UTF-8 bytes，无法安全裁剪")
        return content


def _backoff(attempt: int) -> float:
    return (2.0, 4.0, 8.0)[min(attempt, 2)]


# --------------------------------------------------------------------------
# 通知字段格式化
# --------------------------------------------------------------------------

# 异常详情单字段上限：超出部分截断，避免长堆栈挤占消息可读性（2048 字节仍由客户端兜底）
MESSAGE_MAX_CHARS = 300


def format_duration(seconds: float) -> str:
    """把秒数格式化为紧凑的人类可读时长（如 42s、5m03s、1h02m03s）。"""
    if seconds < 0:
        raise ValueError(f"耗时不得为负: {seconds!r}")
    total = int(seconds + 0.5)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_metric(name: str, value: Any) -> str:
    """格式化单个指标副本：`name=value`（浮点按 6 位有效数字）。"""
    return f"{name}={_format_number(value)}"


def format_metric_summary(summary: Mapping[str, Any] | None, metric_name: str | None = None) -> str:
    """从 run summary 提炼通知指标：期望指标最优值 + 对应 test 指标。

    selection 记录验证集最优值，展示为 `期望指标名=值@epoch=N`（stage_metrics 键名
    自带 stage 前缀）；test 阶段同名指标是训练成果的最终评价，存在即附带。
    没有 selection 时回退 test/val 阶段全部指标（稀有路径）。
    """
    if not summary:
        return ""
    selection = summary.get("selection") or {}
    best_value = selection.get("best_value")
    stage_metrics = summary.get("stage_metrics") or {}
    if best_value is not None:
        parts = []
        best = format_metric(metric_name or "best", best_value)
        best_epoch = selection.get("best_epoch")
        parts.append(best if best_epoch is None else f"{best}@epoch={best_epoch}")
        if metric_name and metric_name.startswith("val/"):
            test_name = f"test/{metric_name.removeprefix('val/')}"
            test_value = (stage_metrics.get("test") or {}).get(test_name)
            if test_value is not None:
                parts.append(format_metric(test_name, test_value))
        return " ".join(parts)
    for stage in ("test", "val"):
        stage_values = stage_metrics.get(stage) or {}
        if stage_values:
            return " ".join(format_metric(name, value) for name, value in sorted(stage_values.items()))
    return ""


def _normalize_message(text: Any) -> str:
    """异常详情扁平化为单行并截断；调用方应传入已脱敏文本。"""
    flat = " ".join(str(text).split())
    if len(flat) <= MESSAGE_MAX_CHARS:
        return flat
    return flat[:MESSAGE_MAX_CHARS] + "…"


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# --------------------------------------------------------------------------
# 生命周期事件模板
# --------------------------------------------------------------------------

_EVENT_TEMPLATES: dict[str, str] = {
    "RUN_STARTED": "[训练开始] run={run_id}",
    "RUN_SUCCEEDED": "[训练成功] run={run_id} {metrics} 耗时={elapsed}",
    "RUN_PREEMPTED": "[训练暂停] run={run_id} {metrics} 耗时={elapsed} 恢复检查点={checkpoint}",
    "RUN_FAILED": "[训练失败] run={run_id} epoch={epoch} step={step} 异常={error_type} 详情={message}",
    "SWEEP_STARTED": "[Sweep开始] sweep={sweep_id} trials={trials} 对比指标={comparison}({mode})",
    "TRIAL_STARTED": "[Trial开始] sweep={sweep_id} trial={trial} 进度={progress}",
    "TRIAL_SUCCEEDED": "[Trial成功] sweep={sweep_id} trial={trial} 进度={progress} {metrics}",
    "TRIAL_PREEMPTED": "[Trial暂停] sweep={sweep_id} trial={trial} 进度={progress} 恢复检查点={checkpoint}",
    "TRIAL_FAILED": ("[Trial失败] sweep={sweep_id} trial={trial} 进度={progress} "
                     "epoch={epoch} 异常={error_type} 详情={message}"),
    "SWEEP_SUCCEEDED": "[Sweep成功] sweep={sweep_id} best={best} {metrics}",
    "SWEEP_PREEMPTED": "[Sweep暂停] sweep={sweep_id} 进度={progress} 暂停于={trial} 恢复检查点={checkpoint}",
    "SWEEP_FAILED": ("[Sweep失败] sweep={sweep_id} trial={trial} 退出码={exit_code} 进度={progress} "
                     "异常={error_type} 详情={message}"),
}


def render_event_template(event: str, **fields: Any) -> str:
    """渲染固定事件模板；关键身份字段（event/status/scope id/error type）不可裁掉。

    非关键可选字段（elapsed/metrics/checkpoint 等）缺失时连同标签一起省略，
    不产生 "指标=" 这类空残段；异常详情（message）扁平化为单行并截断。
    """
    if event not in _EVENT_TEMPLATES:
        raise WecomError(f"未知事件模板: {event!r}")
    template = _EVENT_TEMPLATES[event]
    required = _required_fields(event)
    for key in required:
        if fields.get(key) in (None, ""):
            raise WecomError(f"事件 {event} 缺少关键字段 {key!r}")
    safe = {k: ("" if v is None else v) for k, v in fields.items()}
    if safe.get("message"):
        safe["message"] = _normalize_message(safe["message"])
    if safe.get("metrics"):
        safe["metrics"] = " ".join(str(safe["metrics"]).split())
    # 模板中未提供的非关键字段补空串
    for key in re.findall(r"\{(\w+)\}", template):
        safe.setdefault(key, "")
    # 空可选字段：标签与占位符一起移除，避免消息出现无信息量的残段
    for key, value in safe.items():
        if key in required or value != "":
            continue
        template = re.sub(rf"(?<!\S)[^\s{{}}]*\{{{re.escape(key)}\}}", "", template)
    return re.sub(r" {2,}", " ", template.format(**safe)).strip()


def _required_fields(event: str) -> list[str]:
    if event.startswith("RUN_"):
        return ["run_id"]
    if event.startswith("TRIAL_"):
        return ["sweep_id", "trial"]
    if event.startswith("SWEEP_"):
        return ["sweep_id"]
    return []
