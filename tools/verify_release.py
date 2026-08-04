#!/usr/bin/env python
"""完整本地发布门禁：Secret scan → pytest+coverage → 双阈值 → wheel → 干净安装 → OpenSpec strict。

任一步失败立即抛 CalledProcessError；使用 sys.executable，文本 subprocess UTF-8，不使用 shell。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd: list[str], cwd: str = REPO, env_extra: dict[str, str] | None = None) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", env=env)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(proc.stdout[-400:] if proc.stdout else f"OK: {' '.join(cmd)}")


def _openspec_cmd() -> list[str]:
    """解析 openspec 可执行命令，不用 shell。

    Windows 下 npm 全局 shim 是 .cmd，CreateProcess 无法直接运行；
    展开为 `node <npm-global>/node_modules/@fission-ai/openspec/bin/openspec.js`。
    """
    import shutil

    shim = shutil.which("openspec")
    if not shim:
        raise RuntimeError("openspec 未安装（npm 全局），无法执行 strict 校验")
    if os.name == "nt" and shim.lower().endswith((".cmd", ".bat")):
        node = shutil.which("node")
        js = os.path.join(os.path.dirname(shim), "node_modules",
                          "@fission-ai", "openspec", "bin", "openspec.js")
        if not (node and os.path.exists(js)):
            raise RuntimeError(f"无法解析 openspec node 入口: {js}")
        return [node, js]
    return [shim]


def _openspec_change_name() -> str:
    """动态发现 openspec/changes 下唯一的活跃变更目录名（排除 archive）。"""
    changes_dir = os.path.join(REPO, "openspec", "changes")
    dirs = [
        d for d in os.listdir(changes_dir)
        if os.path.isdir(os.path.join(changes_dir, d))
        and not d.startswith(".")
        and os.path.isfile(os.path.join(changes_dir, d, "proposal.md"))
    ]
    if len(dirs) != 1:
        raise RuntimeError(f"openspec/changes 应恰好一个活跃变更，得到 {dirs}")
    return dirs[0]


def main() -> int:
    # 1. Secret scan
    run([PY, os.path.join("tools", "scan_secrets.py")])
    # 2. 完整 pytest
    run([PY, "-m", "pytest", "-q", "tests/"])
    # 2b. CPU 多进程门禁（OSR-011）：屏蔽 CUDA，强制 gloo/launcher 实际执行
    run([PY, "-m", "pytest", "-q", "tests/distributed/test_gloo_training.py",
         "tests/training/test_launcher.py"],
        env_extra={"DLH_ALLOW_MP": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    # 3. coverage 双阈值（check_coverage 内部预导入重依赖并测量）
    run([PY, os.path.join("tools", "check_coverage.py")])
    # 4. wheel 构建
    dist = os.path.join(tempfile.mkdtemp(), "dist")
    os.makedirs(dist, exist_ok=True)
    run([PY, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist])
    # 5. 干净安装审计
    run([PY, os.path.join("tools", "verify_clean_install.py"), dist])
    # 6. OpenSpec strict
    run(_openspec_cmd() + ["validate", _openspec_change_name(), "--strict", "--no-interactive"])
    print("发布门禁全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
