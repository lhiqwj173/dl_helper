"""离线 HTML 报告：只读 Artifact，HTML escape，相对 PNG，幂等。

不导入用户模型或数据代码；Matplotlib figure 关闭；重复生成覆盖同名文件。
"""
from __future__ import annotations

import html
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import read_json

_REPORT_VERSION = "1.0.0"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _stage_metrics(metrics_jsonl_path: str) -> dict[str, dict[str, Any]]:
    """聚合 metrics.jsonl 每个 stage 最后一条记录的 metrics 与 extended。"""
    out: dict[str, dict[str, Any]] = {}
    if not os.path.exists(metrics_jsonl_path):
        return out
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = record.get("stage")
            if stage is None:
                continue
            out[stage] = record
    return out


def _safe_path(run_dir: str, rel: str) -> str:
    real = os.path.realpath(os.path.join(run_dir, rel))
    if not (real == os.path.realpath(run_dir) or real.startswith(os.path.realpath(run_dir) + os.sep)):
        raise ValueError(f"报告引用逃逸 run 目录: {rel!r}")
    return real


def _read_predictions_arrays(run_dir: str, split: str, limit: int) -> dict[str, np.ndarray]:
    """读取 split 的预测分片并拼接为数组。"""
    pred_dir = os.path.join(run_dir, "predictions", split)
    manifest_path = os.path.join(pred_dir, "prediction-manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    manifest = read_json(manifest_path)
    arrays: dict[str, list[np.ndarray]] = {}
    for shard in manifest.get("shards", []):
        file_name = shard["file"]
        shard_path = _safe_path(pred_dir, file_name)
        data = np.load(shard_path, allow_pickle=False)
        for field in shard.get("fields", {}):
            arrays.setdefault(field, []).append(data[field])
    out: dict[str, np.ndarray] = {}
    for field, parts in arrays.items():
        if parts:
            out[field] = np.concatenate(parts)[:limit]
    return out


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
    return f"<table class='table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_metric_table(summary: Mapping[str, Any], stage_metrics: Mapping[str, dict[str, Any]]) -> str:
    """逐 stage 展示全量指标原始值。"""
    sections = ""
    for stage, record in sorted(stage_metrics.items()):
        metrics = record.get("metrics", {})
        rows = [[k, v] for k, v in sorted(metrics.items())]
        sections += f"<h3>Stage: {_esc(stage)}</h3>" + _render_table(["metric", "value"], rows)
    return sections


def _render_confusion_image(run_dir: str, assets_dir: str, stage: str, extended: Mapping[str, Any]) -> str:
    """生成混淆矩阵 PNG（相对路径）。"""
    key = f"{stage}/confusion_weighted"
    confusion = extended.get(key)
    if confusion is None:
        # 从 prediction arrays 计算
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.asarray(confusion)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({stage})")
        fig.colorbar(im)
        os.makedirs(assets_dir, exist_ok=True)
        name = f"confusion-{stage}.png"
        path = os.path.join(assets_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return f"<img src='assets/{_esc(name)}' alt='confusion {_esc(stage)}'/>"
    except Exception:
        return ""


def _render_regression_scatter(run_dir: str, assets_dir: str, stage: str) -> str:
    arrays = _read_predictions_arrays(run_dir, stage, 5000)
    if "targets" not in arrays or "predictions" not in arrays:
        return ""
    t = np.asarray(arrays["targets"]).reshape(-1)
    p = np.asarray(arrays["predictions"]).reshape(-1)
    if t.shape[0] != p.shape[0] or t.shape[0] == 0:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(t, p, s=4, alpha=0.3)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"Predicted vs Actual ({stage})")
        os.makedirs(assets_dir, exist_ok=True)
        name = f"scatter-{stage}.png"
        fig.savefig(os.path.join(assets_dir, name), bbox_inches="tight")
        plt.close(fig)
        return f"<img src='assets/{_esc(name)}' alt='scatter {_esc(stage)}'/>"
    except Exception:
        return ""


def generate_run_report(run_dir: str, out_dir: str | None = None) -> str:
    """从 run Artifact 生成离线 HTML 报告，返回 index.html 路径。幂等。"""
    run_dir = os.path.realpath(run_dir)
    if not os.path.exists(os.path.join(run_dir, "run-manifest.json")) and not os.path.exists(
        os.path.join(run_dir, "pause-manifest.json")
    ):
        # 允许暂停报告；无终态时仍可读 summary
        pass
    summary_path = os.path.join(run_dir, "metrics", "summary.json")
    summary: dict[str, Any] = read_json(summary_path) if os.path.exists(summary_path) else {}
    stage_metrics = _stage_metrics(os.path.join(run_dir, "metrics", "metrics.jsonl"))

    report_dir = out_dir or os.path.join(run_dir, "report")
    assets_dir = os.path.join(report_dir, "assets")
    os.makedirs(report_dir, exist_ok=True)

    # 通用上下文
    backend = summary.get("backend", "unknown")
    status = summary.get("status", "unknown")
    run_id = summary.get("run_id", os.path.basename(run_dir))
    images_html = ""

    # 任务页（从 summary 的 model_signature 与 extended 判断）
    report_kind = "general"
    ext = _collect_extended(stage_metrics)
    if any(any(k.endswith("/confusion_weighted") for k in rec) for rec in ext.values()):
        report_kind = "multiclass"
    elif any(any(k.endswith("/per_label") for k in rec) for rec in ext.values()):
        report_kind = "multilabel"
    elif any(any(k.endswith("/per_target") for k in rec) for rec in ext.values()):
        report_kind = "regression"

    if report_kind == "multiclass":
        for stage, rec in ext.items():
            images_html += _render_confusion_image(run_dir, assets_dir, stage, rec)
    elif report_kind == "regression":
        for stage in stage_metrics:
            images_html += _render_regression_scatter(run_dir, assets_dir, stage)

    body = f"""
<h1>Run Report: {_esc(run_id)}</h1>
<div class='meta'>
<p>backend: {_esc(backend)} | status: {_esc(status)} | report_version: {_esc(_REPORT_VERSION)}</p>
<p>run_dir: {_esc(run_dir)}</p>
</div>
{_render_context_table(summary)}
<h2>Metrics</h2>
{_render_metric_table(summary, stage_metrics)}
<h2>Figures</h2>
{images_html}
"""
    css = """
body { font-family: sans-serif; margin: 2em; }
h1 { color: #1f4e79; }
table.table { border-collapse: collapse; margin: 1em 0; }
table.table th, table.table td { border: 1px solid #ccc; padding: 4px 8px; }
table.table th { background: #eef3fa; }
.meta p { color: #555; }
img { max-width: 600px; margin: 0.5em; }
"""
    index_path = os.path.join(report_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Run {_esc(run_id)}</title>"
                f"<style>{css}</style></head><body>{body}</body></html>")
    return index_path


def _collect_extended(stage_metrics: Mapping[str, dict[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {stage: rec.get("extended", {}) for stage, rec in stage_metrics.items()}


def _render_context_table(summary: Mapping[str, Any]) -> str:
    rows: list[list[Any]] = []
    for key in ("schema_version", "run_id", "backend", "status", "epoch", "global_step",
                "config_fingerprint", "tuning_fingerprint", "data_fingerprint"):
        if key in summary:
            rows.append([key, summary[key]])
    selection = summary.get("selection")
    if isinstance(selection, Mapping):
        for k, v in selection.items():
            rows.append([f"selection.{k}", v])
    if not rows:
        return ""
    return "<h2>Context</h2>" + _render_table(["field", "value"], rows)


def generate_sweep_report(sweep_dir: str, out_dir: str | None = None) -> str:
    """从 sweep Artifact 生成聚合 HTML 报告。只读、幂等。"""
    sweep_dir = os.path.realpath(sweep_dir)
    report_dir = out_dir or os.path.join(sweep_dir, "sweep-report")
    os.makedirs(report_dir, exist_ok=True)
    manifest = {}
    manifest_path = os.path.join(sweep_dir, "sweep-manifest.json")
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
    ranking = manifest.get("ranking", [])
    rows = []
    if isinstance(ranking, list):
        for entry in ranking:
            if isinstance(entry, Mapping):
                rows.append([
                    entry.get("rank", ""),
                    entry.get("trial", entry.get("run_id", "")),
                    entry.get("value", ""),
                ])
    table = _render_table(["rank", "trial", "comparison value"], rows) if rows else "<p>无排名（未成功或未排名）</p>"
    best = manifest.get("best_trial")
    best_html = f"<p>best trial: {_esc(best)}</p>" if best else ""
    body = f"<h1>Sweep Report: {_esc(os.path.basename(sweep_dir))}</h1>{best_html}<h2>Ranking</h2>{table}"
    css = "body { font-family: sans-serif; margin: 2em; } table { border-collapse: collapse; } table td, table th { border: 1px solid #ccc; padding: 4px 8px; }"
    index_path = os.path.join(report_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>")
    return index_path
