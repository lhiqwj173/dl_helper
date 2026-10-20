"""数据集相对文件清单；只比较名称和划分，不读取样本内容。"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

DATASET_FILES = "dataset-files.json"


def validate_dataset_files(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"schema_version", "files", "train_files", "validation_files"}:
        raise ValueError("数据文件清单必须包含 schema_version、files、train_files、validation_files")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("数据文件清单版本不兼容")
    for key in ("files", "train_files", "validation_files"):
        names = value[key]
        if not isinstance(names, list) or not names:
            raise ValueError(f"数据文件清单 {key} 必须是非空列表")
        for name in names:
            if not isinstance(name, str) or not name or "\\" in name:
                raise ValueError(f"数据文件清单含非法名称：{name!r}")
            path = PurePosixPath(name)
            if (not path.parts or path.is_absolute() or ":" in path.parts[0]
                    or str(path) != name or any(p in (".", "..") for p in path.parts)):
                raise ValueError(f"数据文件清单必须使用相对路径：{name!r}")
        if names != sorted(set(names)):
            raise ValueError(f"数据文件清单 {key} 必须排序且无重复")
    if not set(value["train_files"]) <= set(value["files"]):
        raise ValueError("训练文件不属于数据文件清单")
    if set(value["train_files"]) & set(value["validation_files"]):
        raise ValueError("训练与验证文件清单发生重叠")
    return value


def read_dataset_files(path: Path) -> dict:
    return validate_dataset_files(json.loads(Path(path).read_text(encoding="utf-8")))


def compare_dataset_files(previous: dict, current: dict) -> None:
    validate_dataset_files(previous)
    validate_dataset_files(current)
    differences = []
    for key in ("files", "train_files", "validation_files"):
        added = sorted(set(current[key]) - set(previous[key]))
        removed = sorted(set(previous[key]) - set(current[key]))
        if added or removed:
            differences.append(f"{key}: 新增 {len(added)} {added[:5]}；缺失 {len(removed)} {removed[:5]}")
    if differences:
        raise ValueError("数据文件清单与检查点不同：\n" + "\n".join(differences))
