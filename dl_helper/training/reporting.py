"""离线 HTML 报告：只读 Artifact，HTML escape，相对 PNG，幂等。

不导入用户模型或数据代码；Matplotlib figure 关闭；重复生成覆盖同名文件。
"""
from __future__ import annotations

import html
import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_text, read_json, write_json

_REPORT_VERSION = "1.0.0"
PROGRESS_REPORT_SCHEMA_VERSION = 1
PROGRESS_HTML = "progress/index.html"
PROGRESS_SNAPSHOT = "progress/progress-snapshot.json"

# 组键集合与 Task 实际产出的标量指标键精确对应（task.py _multiclass/_multilabel/
# _regression_definitions）；渲染时仅绘制 metrics.jsonl 中真实出现的键
_TASK_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multiclass", ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                    "f1_macro", "f1_weighted")),
    ("multilabel", ("subset_accuracy", "hamming_loss", "precision_macro", "precision_micro",
                    "recall_macro", "recall_micro", "f1_macro", "f1_micro", "f1_weighted")),
    ("regression", ("mae", "mse", "r2", "r2_variance_weighted")),
)

# 各任务组在 extended 结构中的识别后缀（design：按 metric key 与 extended 结构共同判定）
_GROUP_EXTENDED_SUFFIXES: dict[str, str] = {
    "multiclass": "confusion_weighted",
    "multilabel": "per_label",
    "regression": "per_target",
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


_PROGRESS_LR_SOURCES = ("optimizer", "estimator_history", "estimator_config", "unavailable")


def validate_progress_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """校验 checkpoint 进度快照合同（fail-fast，非法输入直接抛 ValueError）。"""
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"progress snapshot 必须为 Mapping: {type(snapshot).__name__}")
    if snapshot.get("schema_version") != PROGRESS_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"progress snapshot schema_version 不兼容: {snapshot.get('schema_version')!r}"
            f" (期望 {PROGRESS_REPORT_SCHEMA_VERSION})"
        )
    backend = snapshot.get("backend")
    if backend not in ("torch", "sklearn"):
        raise ValueError(f"progress snapshot backend 非法: {backend!r}")
    for key in ("run_id", "checkpoint_id"):
        if not isinstance(snapshot.get(key), str) or not snapshot[key]:
            raise ValueError(f"progress snapshot 缺少非空 {key!r}")
    position = snapshot.get("position")
    if not isinstance(position, Mapping):
        raise ValueError("progress snapshot 缺少 position")
    for key in ("epoch", "batch_in_epoch", "global_step"):
        value = position.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"position.{key} 必须为非负 int: {value!r}")
    max_epochs = snapshot.get("max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs < 1:
        raise ValueError(f"max_epochs 必须为正 int: {max_epochs!r}")
    partial = snapshot.get("partial_epoch")
    if partial is not None:
        if not isinstance(partial, Mapping):
            raise ValueError(f"partial_epoch 必须为 Mapping 或 null: {type(partial).__name__}")
        if partial.get("epoch") != position["epoch"]:
            raise ValueError("partial_epoch.epoch 必须与 position.epoch 一致")
        for key in ("train_loss", "learning_rate"):
            value = partial.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"partial_epoch.{key} 必须为有限数值: {value!r}")
    lr = snapshot.get("learning_rate")
    if not isinstance(lr, Mapping):
        raise ValueError("progress snapshot 缺少 learning_rate")
    if lr.get("source") not in _PROGRESS_LR_SOURCES:
        raise ValueError(f"learning_rate.source 非法: {lr.get('source')!r}")
    return dict(snapshot)


def _progress_history(metrics_jsonl_path: str) -> dict[str, Any]:
    """解析 metrics.jsonl 全量记录为 per-epoch 曲线数据。

    返回 {"loss": {stage: {epoch: value}}, "scalars": {name: {stage: {epoch: value}}},
    "learning_rate": {epoch: lr}, "extended_keys": set[str]}。loss 键不进入 scalars。
    """
    loss: dict[str, dict[int, float]] = {}
    scalars: dict[str, dict[str, dict[int, float]]] = {}
    learning_rate: dict[int, float] = {}
    extended_keys: set[str] = set()
    if not os.path.exists(metrics_jsonl_path):
        return {"loss": loss, "scalars": scalars, "learning_rate": learning_rate,
                "extended_keys": extended_keys}
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            # fail-fast：损坏/截断记录静默跳过会误导进度判断，必须中止报告生成
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"metrics.jsonl 第 {line_number} 行无法解析，拒绝生成不完整进度报告: "
                    f"{metrics_jsonl_path}"
                ) from exc
            stage = record.get("stage")
            epoch = record.get("epoch")
            if not isinstance(stage, str) or not isinstance(epoch, int) or isinstance(epoch, bool):
                # 身份字段缺失说明记录非本系统写入或已损坏，静默丢弃会缺曲线点
                raise ValueError(
                    f"metrics.jsonl 第 {line_number} 行缺少合法 stage/epoch 身份字段，"
                    f"拒绝生成不完整进度报告: {metrics_jsonl_path}"
                )
            prefix = f"{stage}/"
            for key, value in record.get("metrics", {}).items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key == f"{stage}/loss":
                    loss.setdefault(stage, {})[epoch] = float(value)
                    continue
                if key.startswith(prefix):
                    scalars.setdefault(key[len(prefix):], {}).setdefault(stage, {})[epoch] = float(value)
            lr_value = record.get("learning_rate")
            if isinstance(lr_value, (int, float)) and not isinstance(lr_value, bool):
                learning_rate[epoch] = float(lr_value)
            extended = record.get("extended", {})
            if isinstance(extended, Mapping):
                extended_keys.update(extended.keys())
    return {"loss": loss, "scalars": scalars, "learning_rate": learning_rate,
            "extended_keys": extended_keys}


def _active_progress_groups(history: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """按标量 metric 与 extended 结构判定活跃任务组；仅返回含标量曲线的组。

    precision_macro/f1_macro 等聚合键同时存在于 multiclass 与 multilabel 定义，
    由 extended 后缀（confusion_weighted/per_label/per_target）唯一判别任务归属：
    恰有一组命中后缀时共享键归属该组，其余组只保留独占键，避免同一曲线重复渲染。
    """
    extended_keys: set[str] = set(history["extended_keys"])
    hits: dict[str, list[str]] = {
        group_name: [name for name in names if name in history["scalars"]]
        for group_name, names in _TASK_METRIC_GROUPS
    }
    extended_owner: str | None = None
    for group_name, _ in _TASK_METRIC_GROUPS:
        suffix = _GROUP_EXTENDED_SUFFIXES[group_name]
        if any(key.endswith(f"/{suffix}") for key in extended_keys):
            if extended_owner is not None:
                raise ValueError(f"metrics history 同时包含多个任务组 extended 结构: {extended_owner} 与 {group_name}")
            extended_owner = group_name
    out: list[tuple[str, list[str]]] = []
    for group_name, _ in _TASK_METRIC_GROUPS:
        names = hits[group_name]
        if extended_owner is not None and group_name != extended_owner:
            owned = set(hits[extended_owner])
            names = [name for name in names if name not in owned]
        if names:
            out.append((group_name, names))
    return out


def _epoch_axis_limit(max_epochs: int) -> int:
    """固定 x 轴为 1..max_epochs；max_epochs==1 时保底可视宽度。"""
    return max_epochs if max_epochs > 1 else 2


def _fig_png_bytes(fig: Any) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _embed_png(fig: Any) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(_fig_png_bytes(fig)).decode("ascii")


def _lr_view_from_snapshot(lr: Mapping[str, Any]) -> dict[str, Any]:
    """snapshot.learning_rate → 主图学习率渲染视图。"""
    source = lr["source"]
    if source == "optimizer":
        return {"mode": "epoch_history"}
    if source == "estimator_history":
        return {"mode": "iter_history", "history": list(lr.get("history") or [])}
    if source == "estimator_config":
        return {"mode": "config", "param_name": lr.get("param_name"),
                "value": float(lr["config_value"])}
    return {"mode": "na"}


def _lr_view_from_history(history: Mapping[str, Any]) -> dict[str, Any]:
    """run report：仅依据落盘 metrics.jsonl 推断学习率可见性，不伪造数值。"""
    if history["learning_rate"]:
        return {"mode": "epoch_history"}
    return {"mode": "na"}


def read_max_epochs(run_dir: str) -> int:
    """从落盘 config.resolved.yaml 读取固定 x 轴上限；缺失或非法即抛错（fail-fast）。"""
    import yaml

    path = os.path.join(run_dir, "config.resolved.yaml")
    if not os.path.exists(path):
        raise ValueError(f"缺少 config.resolved.yaml，无法确定固定 epoch 轴上限: {run_dir}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    value = (data or {}).get("training", {}).get("max_epochs")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"training.max_epochs 必须为正 int: {value!r}")
    return value


def _render_trend_figure(history: Mapping[str, Any], max_epochs: int, lr_view: Mapping[str, Any],
                         partial: Mapping[str, Any] | None = None) -> str:
    """主图：固定 1..max_epochs 轴上的 train/val loss 与学习率；返回内嵌 PNG data URI。

    run report 与 checkpoint progress report 共用；partial 仅中途 checkpoint 快照提供。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = _epoch_axis_limit(max_epochs)

    fig, ax_loss = plt.subplots(figsize=(7.0, 3.6), dpi=110)
    ax_lr = ax_loss.twinx()
    stage_styles = {"train": "tab:blue", "val": "tab:orange"}
    for stage, color in stage_styles.items():
        points = sorted(history["loss"].get(stage, {}).items())
        if points:
            ax_loss.plot([e + 1 for e, _ in points], [v for _, v in points],
                         marker="o", ms=3, color=color, label=f"{stage} loss")
    if partial is not None:
        ax_loss.plot([partial["epoch"] + 1], [partial["train_loss"]], marker="D", ms=6,
                     color="tab:red", linestyle="none", label="partial train loss")

    mode = lr_view["mode"]
    lr_label: str | None = None
    if mode == "epoch_history":
        points = sorted(history["learning_rate"].items())
        if points:
            ax_lr.plot([e + 1 for e, _ in points], [v for _, v in points],
                       marker=".", ms=3, color="tab:green", linestyle="--",
                       label="learning rate (optimizer)")
            lr_label = "learning rate (optimizer)"
        if partial is not None:
            ax_lr.plot([partial["epoch"] + 1], [partial["learning_rate"]], marker="D", ms=6,
                       color="darkgreen", linestyle="none", label="partial learning rate")
            lr_label = "learning rate (optimizer)"
    elif mode == "iter_history":
        hist = lr_view.get("history") or []
        if hist:
            xs = [1.0 + (limit - 1) * i / max(len(hist) - 1, 1) for i in range(len(hist))]
            ax_lr.plot(xs, [float(v) for _, v in hist], color="tab:green", linestyle="--",
                       label="learning rate (estimator history, uniform on epoch axis)")
            lr_label = "learning rate (estimator history)"
    elif mode == "config":
        value = float(lr_view["value"])
        param = lr_view.get("param_name") or "learning_rate"
        ax_lr.axhline(value, color="tab:green", linestyle=":", linewidth=1.5,
                      label=f"learning rate ({param} init)")
        lr_label = f"learning rate ({param} init)"
    else:
        ax_lr.text(0.98, 0.95, "learning rate: N/A", transform=ax_lr.transAxes,
                   ha="right", va="top", color="gray", fontsize=10)

    ax_loss.set_xlim(1, limit)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    if lr_label is not None:
        ax_lr.set_ylabel("learning rate")
    if any(a.get_legend_handles_labels()[0] for a in (ax_loss, ax_lr)):
        lines, labels = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_lr.get_legend_handles_labels()
        ax_loss.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    ax_loss.set_title("Loss / learning rate (fixed epoch axis)")
    fig.tight_layout()
    try:
        return _embed_png(fig)
    finally:
        plt.close(fig)


def _render_progress_main_figure(history: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    return _render_trend_figure(history, int(snapshot["max_epochs"]),
                                _lr_view_from_snapshot(snapshot["learning_rate"]),
                                partial=snapshot.get("partial_epoch"))


def _render_progress_group_figure(group_name: str, metric_names: Sequence[str],
                                  history: Mapping[str, Any], max_epochs: int) -> str:
    """任务组副图：每标量指标独立子图，train/val 共轴；返回内嵌 PNG data URI。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = _epoch_axis_limit(max_epochs)
    ncols = min(len(metric_names), 2)
    nrows = math.ceil(len(metric_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             dpi=110, squeeze=False)
    stage_styles = {"train": "tab:blue", "val": "tab:orange"}
    flat_axes = [ax for row in axes for ax in row]
    for ax, name in zip(flat_axes, metric_names):
        for stage, color in stage_styles.items():
            points = sorted(history["scalars"][name].get(stage, {}).items())
            if points:
                ax.plot([e + 1 for e, _ in points], [v for _, v in points],
                        marker="o", ms=3, color=color, label=f"{stage}")
        ax.set_xlim(1, limit)
        ax.set_xlabel("epoch")
        ax.set_title(name)
        ax.legend(fontsize=8)
    for ax in flat_axes[len(metric_names):]:
        ax.set_visible(False)
    fig.suptitle(f"Task metrics: {group_name}")
    fig.tight_layout()
    try:
        return _embed_png(fig)
    finally:
        plt.close(fig)


def generate_checkpoint_progress_report(run_dir: str, snapshot: Mapping[str, Any],
                                        progress_dir: str) -> str:
    """从 run Artifact 历史与显式 snapshot 生成 checkpoint 内嵌进度报告。

    写 ``progress_dir/index.html`` 与 ``progress_dir/progress-snapshot.json``，
    返回 index.html 路径。只读取已落盘 Artifact（metrics/metrics.jsonl）；Matplotlib
    渲染失败直接抛异常，不产出空报告；重复生成覆盖同名文件（幂等）。
    """
    snapshot = validate_progress_snapshot(snapshot)
    history = _progress_history(os.path.join(run_dir, "metrics", "metrics.jsonl"))
    os.makedirs(progress_dir, exist_ok=True)

    main_png = _render_progress_main_figure(history, snapshot)
    group_pngs = [
        (group_name, names,
         _render_progress_group_figure(group_name, names, history, int(snapshot["max_epochs"])))
        for group_name, names in _active_progress_groups(history)
    ]

    position = snapshot["position"]
    partial = snapshot.get("partial_epoch")
    meta_rows = [
        ["backend", snapshot["backend"]],
        ["checkpoint_id", snapshot["checkpoint_id"]],
        ["epoch", position["epoch"]],
        ["batch_in_epoch", position["batch_in_epoch"]],
        ["global_step", position["global_step"]],
        ["max_epochs", snapshot["max_epochs"]],
        ["created_utc", snapshot.get("created_utc", "")],
    ]
    if partial is not None:
        meta_rows += [
            ["partial.epoch", partial["epoch"]],
            ["partial.train_loss", partial["train_loss"]],
            ["partial.learning_rate", partial["learning_rate"]],
        ]
    lr = snapshot["learning_rate"]
    lr_rows: list[list[Any]] = [["source", lr["source"]]]
    if lr["source"] == "unavailable":
        # N/A 必须在 HTML 文本中可检索，不能只出现在渲染图内部
        lr_rows.append(["learning_rate", "N/A"])
    if lr.get("param_name") is not None:
        lr_rows.append(["param_name", lr["param_name"]])
    if lr.get("config_value") is not None:
        lr_rows.append(["config_value", lr["config_value"]])
    if lr.get("history"):
        lr_rows.append(["history_points", len(lr["history"])])

    body = f"""
<h1>Checkpoint Progress Report: {_esc(snapshot['run_id'])}/{_esc(snapshot['checkpoint_id'])}</h1>
<div class='meta'><p>position: epoch {_esc(position['epoch'])} batch {_esc(position['batch_in_epoch'])}
step {_esc(position['global_step'])} / max_epochs {_esc(snapshot['max_epochs'])}</p></div>
<h2>Position</h2>
{_render_table(['field', 'value'], meta_rows)}
<h2>Main trend</h2>
<img src='{main_png}' alt='loss and learning rate'/>
<h2>Learning rate</h2>
{_render_table(['field', 'value'], lr_rows)}
<h2>Task metrics</h2>
{(''.join(f"<p>{_esc(group_name)} subplots: {_esc(', '.join(metric_names))}</p>"
          f"<img src='{png}' alt='metrics {_esc(group_name)}'/>"
          for group_name, metric_names, png in group_pngs)
  or '<p>当前任务无可用标量指标组</p>')}
"""
    css = """
body { font-family: sans-serif; margin: 2em; }
h1 { color: #1f4e79; }
table.table { border-collapse: collapse; margin: 1em 0; }
table.table th, table.table td { border: 1px solid #ccc; padding: 4px 8px; }
table.table th { background: #eef3fa; }
.meta p { color: #555; }
img { max-width: 760px; margin: 0.5em; display: block; }
"""
    index_path = os.path.join(progress_dir, "index.html")
    atomic_write_text(
        index_path,
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Progress {html.escape(str(snapshot['checkpoint_id']))}</title>"
        f"<style>{css}</style></head><body>{body}</body></html>",
    )
    write_json(os.path.join(progress_dir, "progress-snapshot.json"), snapshot)
    return index_path


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
    # 全量 per-epoch 历史（趋势主图与任务副图数据源；_stage_metrics 只保留每 stage 末条）
    stage_metrics_history = _progress_history(os.path.join(run_dir, "metrics", "metrics.jsonl"))

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

    # 固定 1..max_epochs 轴的趋势主图与任务指标副图（与 checkpoint progress report 共用渲染）
    max_epochs = read_max_epochs(run_dir)
    trend_png = _render_trend_figure(stage_metrics_history, max_epochs,
                                     _lr_view_from_history(stage_metrics_history))
    group_pngs = [
        (group_name, names,
         _render_progress_group_figure(group_name, names, stage_metrics_history, max_epochs))
        for group_name, names in _active_progress_groups(stage_metrics_history)
    ]
    trend_section = f"<h2>Trend (fixed epoch axis)</h2><img src='{trend_png}' alt='loss and learning rate'/>"
    group_section = "".join(
        f"<p>{_esc(group_name)} subplots: {_esc(', '.join(metric_names))}</p>"
        f"<img src='{png}' alt='metrics {_esc(group_name)}'/>"
        for group_name, metric_names, png in group_pngs
    )

    body = f"""
<h1>Run Report: {_esc(run_id)}</h1>
<div class='meta'>
<p>backend: {_esc(backend)} | status: {_esc(status)} | report_version: {_esc(_REPORT_VERSION)}</p>
<p>run_dir: {_esc(run_dir)}</p>
</div>
{_render_context_table(summary)}
{trend_section}
<h2>Metrics</h2>
{_render_metric_table(summary, stage_metrics)}
<h2>Figures</h2>
{images_html}
{group_section}
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
