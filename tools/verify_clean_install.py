#!/usr/bin/env python
"""干净安装审计：构建 wheel，仓库外 cwd 以 no-deps 探测新旧 import。

用法：python tools/verify_clean_install.py [dist_dir]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLD_MODULES = [
    "dl_helper.trainer", "dl_helper.tester", "dl_helper.tracker", "dl_helper.train_param",
    "dl_helper.models", "dl_helper.rl", "dl_helper.transforms", "dl_helper.data",
    "dl_helper.scheduler", "dl_helper.tool", "dl_helper.acc",
]


def _run(cmd: list[str], cwd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          env=env or {**os.environ, "PYTHONIOENCODING": "utf-8"})
    return proc


def main() -> int:
    dist_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    # 1. 构建 wheel
    build = _run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist_dir],
                 cwd=REPO)
    if build.returncode != 0:
        print("wheel 构建失败", file=sys.stderr)
        return 1
    wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
    if not wheels:
        print("未找到 wheel", file=sys.stderr)
        return 1
    wheel = os.path.join(dist_dir, wheels[0])

    # 2. 安装到临时 target（no-deps）
    target = tempfile.mkdtemp(prefix="dlh-clean-install-")
    install = _run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", target, wheel],
                   cwd=REPO)
    if install.returncode != 0:
        print("wheel 安装失败", file=sys.stderr)
        return 1

    # 3. 仓库外 cwd 探测：PYTHONPATH 只指向 target，避免 editable/source 污染
    outside = tempfile.mkdtemp(prefix="dlh-outside-")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = target
    env["PYTHONIOENCODING"] = "utf-8"
    probe = _run([sys.executable, "-c",
                  "import dl_helper; import dl_helper.training; import dl_helper.training.cli; print('NEW OK')"],
                 cwd=outside, env=env)
    if probe.returncode != 0:
        print("新 API 导入失败", file=sys.stderr)
        print(probe.stderr, file=sys.stderr)
        return 1

    # 4. 旧 import 负向
    for mod in OLD_MODULES:
        proc = _run([sys.executable, "-c", f"import {mod}"], cwd=outside, env=env)
        if proc.returncode == 0:
            print(f"旧模块 {mod} 意外导入成功", file=sys.stderr)
            return 1
    print("干净安装审计通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
