"""测试夹具：把 examples/ 示例项目加入导入路径与 PYTHONPATH。

示例与真实训练项目使用同一入口：CLI 路径经 `--project-dir <repo>/examples` 使用示例；
直接导入示例模块（单元测试）时由本 conftest 提供同一导入路径，二者都不把示例作为库模块。
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_REPO, "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
_parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
if _EXAMPLES not in _parts:
    os.environ["PYTHONPATH"] = os.pathsep.join([_EXAMPLES] + _parts)