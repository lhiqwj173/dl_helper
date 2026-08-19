# Kaggle 使用指南

本库只提供训练引擎和生命周期服务，不包含你的训练项目。Kaggle Notebook 中应准备三个独立路径：

- `project_dir`：你的模型/数据代码，必须包含 `build_experiment(config)`；
- `config_path`：你的 YAML 配置；
- `output_root`：`/kaggle/working` 下的产物目录（默认 `/kaggle/working/dl-helper-runs`）。

> 本页所有可执行代码都是 Python Notebook 单元：用 `subprocess` 调当前 kernel 的
> `sys.executable`，不依赖系统 shell 续行或开发者本机解释器路径。
> 说明中凡是「运行单元」即指把该代码块粘贴到 Kaggle Notebook 的一个代码单元执行。

## 0. 一个单元助手

每次启动训练前，先定义统一的退出码处理助手，后续单元直接复用它：

```python
import subprocess
import sys

def run_train(params, *, expect=(0, 75)):
    """运行 train；0 成功、75 预算保护暂停，其余退出码一律视为失败。"""
    proc = subprocess.run(
        [sys.executable, "-m", "dl_helper.training.cli", "train"] + params,
        text=True, encoding="utf-8",
    )
    if proc.returncode not in expect:
        raise RuntimeError(f"训练失败，退出码: {proc.returncode}")
    return proc.returncode
```

`0` 表示完成；`75` 表示预算保护暂停（已写入检查点、提交 AList 并生成 `pause-manifest.json`）。
**不要把 75 当作失败**：它和 0 一样是可继续的终态（只有 `75` 才能被同一 run ID 自动恢复）。

## 1. 安装 dl-helper

在 `/kaggle/working` 下 clone 本仓库（或改用你上传的 wheel），然后安装：

```python
import os
import subprocess
import sys

repo_dir = "/kaggle/working/dl-helper"
repo_url = "https://github.com/<you>/dl_helper.git"  # 需替换为你的仓库地址
if os.path.exists(repo_dir):
    raise RuntimeError(f"目录已存在，请新建 Kaggle Session 后重试: {repo_dir}")

# 可选：固定版本 ref（tag/分支/短 SHA）
os.environ.setdefault("DL_HELPER_GIT_REF", "")
# bootstrap 复用手工 clone 的仓库：必须指向该 Git 工作树根，并提供仓库 URL
os.environ["DL_HELPER_REPO_DIR"] = repo_dir
os.environ["DL_HELPER_GIT_REPO"] = repo_url

def checked(argv, *, cwd=None):
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"命令失败，退出码 {proc.returncode}: {argv}")
    return proc

checked(["git", "clone", repo_url, repo_dir])
if os.environ["DL_HELPER_GIT_REF"]:
    checked(["git", "checkout", os.environ["DL_HELPER_GIT_REF"]], cwd=repo_dir)

# 只安装库（并不运行示例训练）；bootstrap 复用上面已 clone 并校验的仓库
checked([sys.executable, f"{repo_dir}/envs/kaggle_bootstrap.py"], cwd=repo_dir)
print("dl-helper 安装完成")
```

`envs/kaggle_bootstrap.py` 只负责 clone（可选 `DL_HELPER_GIT_REF`，支持 tag/分支/短 SHA）和
`pip install -e . --no-deps`，不调用库内示例训练。

## 2. 准备项目目录与配置

你的训练项目放在 `/kaggle/working/my-project`，结构如下：

```
/kaggle/working/my-project/
  my_project.py        # 导出 build_experiment(config)
  configs/
    kaggle.yaml        # 你的 YAML 配置
```

`build_experiment(config)` 返回 `TorchExperiment` 或 `SklearnExperiment`（写法见
[训练指南](guide.md)）。本库不自动发现业务代码或数据：`project_dir`、配置、Experiment 引用都要显式给出。
Kaggle 输入数据必须显式挂在 `/kaggle/input/...`，输出默认落在 `/kaggle/working/dl-helper-runs`。

在 Notebook 里生成配置文件（直接在项目中手工维护 YAML 也可，这里是生成示例）：

```python
from pathlib import Path
import yaml

project_dir = Path("/kaggle/working/my-project")
config_path = project_dir / "configs" / "kaggle.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)

config = {
    "schema_version": 1,
    "run": {"name": "my-project", "id": None, "output_root": None,
            # source_revision 是无空白版本标识（tag/分支/短 SHA/任意标签均可）；
            # Kaggle 的 /kaggle/working 不是 Git 仓库，必须显式提供，
            # 否则版本解析失败会在预检阶段报错。
            "source_revision": "my-project-v1", "seed": 42, "tags": {}},
    "experiment": {"lr": 0.001, "batch_size": 128},
    "training": {"max_epochs": 30, "log_every_steps": 20},
    "backend": {"type": "torch", "torch": {
        "gradient_accumulation_steps": 1, "mixed_precision": "auto", "compile": False,
        "clip_grad_norm": 1.0, "deterministic": "off", "matmul_precision": "high",
        "find_unused_parameters": False}, "sklearn": None},
    "distributed": {"num_processes": "auto"},
    "selection": {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0},
    "checkpoint": {"every_epochs": 1, "every_optimizer_steps": None, "keep_last": 2},
    "report": {"enabled": True, "curve_sample_limit": 100000,
               "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
    "remote": {"type": "alist", "host": "https://your-alist.example.com", "base_path": "/dl-helper/my-project",
               "user_secret_key": "ALIST_USER", "password_secret_key": "ALIST_PWD",
               "connect_timeout_seconds": 10, "read_timeout_seconds": 60,
               "max_attempts": 3, "async_upload": False, "failure_policy": "required"},
    "notifications": {"type": "wecom", "corp_id_secret_key": "WECOM_CORP_ID",
                      "corp_secret_key": "WECOM_CORP_SECRET",
                      "agent_id_secret_key": "WECOM_AGENT_ID", "to_user": "@all",
                      "connect_timeout_seconds": 10, "read_timeout_seconds": 30,
                      "max_attempts": 3, "failure_policy": "required"},
}
config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("配置已写入:", config_path)
```

> 配置**不含** `runtime` 与 `checkpoint.resume`：Kaggle 的运行预算由平台执行策略固定为
> **660 分钟训练 + 10 分钟收尾**（文档/审计见 run 目录 `execution-policy.json`），恢复策略由 CLI
> 内部自动决定或显式 `--resume none|required` 覆盖。上一版本的根级 `runtime` 与 `checkpoint.resume`
> 字段已被删除，写入即报未知字段失败。

## 3. Secrets

Kaggle 运行强制启用 AList 和企业微信，且两者 `failure_policy=required`。在 Kaggle Secrets 中创建
5 个键（本地调试时可用同名环境变量代替）：`ALIST_USER`、`ALIST_PWD`、`WECOM_CORP_ID`、
`WECOM_CORP_SECRET`、`WECOM_AGENT_ID`。运行前预检会逐个读取并报告缺失键；Secret 值不会写入
配置、日志或错误证据。

## 4. 首次训练

```python
run_train([
    "--project-dir", "/kaggle/working/my-project",
    "--config", "/kaggle/working/my-project/configs/kaggle.yaml",
    "--experiment", "my_project:build_experiment",
    "--run-id", "my-project-v1",
])
print("首次训练结束")
```

无需 `--resume`：省略时按内部自动恢复策略执行——本地有最新检查点则恢复，本地没有则从 AList 查询，
两处都无则从头开始。`--resume none` 禁止恢复；`--resume required` 在没有兼容检查点时直接失败。

训练会自动预检（配置、ExecutionPolicy、Secret、数据路径、后端、磁盘、版本和服务），预检失败会聚合
列出全部错误并立即终止，不再有独立的 `doctor` 命令。

## 5. 跨 Session 自动恢复

新 Session 先完成第 1~4 步（重新挂载同一 AList/数据集、安装相同版本库和项目），然后使用**完全相同的
`--run-id`** 再次运行第 4 个单元即可：

```python
run_train([
    "--project-dir", "/kaggle/working/my-project",
    "--config", "/kaggle/working/my-project/configs/kaggle.yaml",
    "--experiment", "my_project:build_experiment",
    "--run-id", "my-project-v1",
])
print("恢复完成")
```

引擎会优先读取本地检查点；本地没有时从 AList 恢复最新检查点。配置、数据身份、模型结构和版本标识
不匹配会直接失败，不会静默从头训练。只有「必须恢复、没有检查点就失败」的审计场景才需要显式
`--resume required`。

## 6. 超参数调优（sweep）

在项目目录外维护 sweep manifest，base 配置和每个 variant 都是普通 YAML。每个 trial 先自动做
零拟合可比性预检，再隔离子进程运行：

```python
run_sweep = subprocess.run(
    [sys.executable, "-m", "dl_helper.training.cli", "sweep",
     "--sweep", "/kaggle/working/my-project/configs/sweep/sweep.yaml",
     "--project-dir", "/kaggle/working/my-project"],
    text=True, encoding="utf-8",
)
# 0 完成、75 预算保护暂停（trial 已保存检查点，可继续）：两者都是受控结果，只对其它非零码抛错
if run_sweep.returncode not in (0, 75):
    raise RuntimeError(f"sweep 失败，退出码: {run_sweep.returncode}")
print("sweep 完成")
```

某个 trial 因预算返回 `75` 时，sweep 会写入暂停清单；下一次 Session 使用相同命令和 `--resume`
继续。全部 trial 成功后执行聚合报告：

```python
sweep_dir = "/kaggle/working/dl-helper-runs/sweeps/<sweep-id>"
report = subprocess.run(
    [sys.executable, "-m", "dl_helper.training.cli", "sweep-report",
     "--sweep-dir", sweep_dir],
    text=True, encoding="utf-8",
)
if report.returncode != 0:
    raise RuntimeError(f"sweep-report 失败，退出码: {report.returncode}")
print("sweep report written")
```

## 7. 本地调试

本地只用于调试时把 `project_dir`、`config`、`output_root` 指向本地路径；AList/企业微信可按配置
关闭（`type: none`）。Local profile 不启用运行预算，不受 Kaggle 660/10 限制；Kaggle 专属的服务
强制与预算策略只在 Kaggle 平台检测到时生效。