"""单机 Torch 检查点压缩、只读 Dataset 发现与严格恢复。"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


def _read_json_bytes(raw: bytes, label: str) -> dict:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{label}")
    return value


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"检查点元数据不是普通文件：{path}")
    return _read_json_bytes(path.read_bytes(), str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"归档成员路径非法：{name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ":" in name or path.as_posix() != name or any(
            part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"归档成员路径非法：{name!r}")
    return path


def _validate_file_inventory(root: Path, expected: dict, *, excluded: set[str]) -> None:
    if not isinstance(expected, dict):
        raise ValueError("checkpoint 文件清单必须是对象")
    # 原始 checkpoint v1 使用宿主 OS 路径；ZIP 外层始终使用 POSIX 路径。
    portable = {name.replace("\\", "/"): metadata for name, metadata in expected.items()}
    if len(portable) != len(expected):
        raise ValueError("checkpoint 文件清单含重复规范路径")
    expected = portable
    for relative, metadata in expected.items():
        _safe_member(relative)
        if not isinstance(metadata, dict) or set(metadata) != {"sha256", "size"}:
            raise ValueError(f"checkpoint 文件元数据非法：{relative}")
        digest = metadata["sha256"]
        size = metadata["size"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in string.hexdigits for character in digest)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ValueError(f"checkpoint 文件校验字段非法：{relative}")
    actual: set[str] = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in subdirectories:
            child = base / name
            if child.is_symlink():
                raise ValueError(f"checkpoint 含符号链接目录：{child}")
        for name in filenames:
            child = base / name
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"checkpoint 含非普通文件：{child}")
            relative = child.relative_to(root).as_posix()
            if relative not in excluded:
                actual.add(relative)
    if actual != set(expected):
        raise ValueError(
            f"checkpoint 文件集合与 manifest 不符；缺失={sorted(set(expected) - actual)} "
            f"多余={sorted(actual - set(expected))}"
        )
    for relative, metadata in expected.items():
        _safe_member(relative)
        if not isinstance(metadata, dict) or set(metadata) != {"sha256", "size"}:
            raise ValueError(f"checkpoint 文件元数据非法：{relative}")
        path = root.joinpath(*_safe_member(relative).parts)
        if path.stat().st_size != metadata["size"] or _sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"checkpoint 文件校验失败：{relative}")


def _validate_latest(run_root: Path, expected_run_id: str) -> tuple[dict, Path, dict]:
    checkpoint_root = run_root / "checkpoints"
    latest = _read_json(checkpoint_root / "latest.json")
    if latest.get("schema_version") != 1:
        raise ValueError("latest.json schema_version 不匹配")
    checkpoint_id = latest.get("checkpoint_id")
    checkpoint_path = latest.get("path")
    if not isinstance(checkpoint_id, str) or checkpoint_path != checkpoint_id:
        raise ValueError("latest.json checkpoint_id/path 非法")
    latest_path = checkpoint_root / "latest.json"
    if latest_path.is_symlink() or not latest_path.is_file():
        raise FileNotFoundError(f"latest.json 不是普通文件：{latest_path}")
    if len(_safe_member(checkpoint_path).parts) != 1:
        raise ValueError("latest.json checkpoint path 必须是单层目录")
    checkpoint_dir = checkpoint_root / checkpoint_path
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"latest checkpoint 目录不存在或非法：{checkpoint_dir}")
    checkpoint_manifest = _read_json(checkpoint_dir / "checkpoint-manifest.json")
    if (checkpoint_manifest.get("schema_version") != 1
            or checkpoint_manifest.get("complete") is not True
            or checkpoint_manifest.get("backend") != "torch"
            or checkpoint_manifest.get("run_id") != expected_run_id
            or checkpoint_manifest.get("checkpoint_id") != checkpoint_id):
        raise ValueError("latest checkpoint manifest 与当前 run 不匹配或未完成")
    _validate_file_inventory(
        checkpoint_dir, checkpoint_manifest.get("files"), excluded={"checkpoint-manifest.json"}
    )
    return latest, checkpoint_dir, checkpoint_manifest


def _config_fingerprint(config_path: Path, expected_run_id: str, expected_source_revision: str) -> str:
    from dl_helper.training.config import config_fingerprint, load_config_file

    config = load_config_file(str(config_path))
    if config.run.source_revision != expected_source_revision:
        raise ValueError("config source_revision 与当前数据包不匹配")
    if config.run.id not in (None, expected_run_id):
        raise ValueError("config run.id 与当前 RUN_ID 不匹配")
    return config_fingerprint(config, resume=True)


def _load_resume_manifest(source: Path) -> tuple[dict, list[str]]:
    manifest_path = source / "resume-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"恢复目录缺少普通文件 resume-manifest.json：{manifest_path}")
    manifest = _read_json(manifest_path)
    return _validate_resume_manifest(manifest)


def _validate_resume_manifest(manifest: dict) -> tuple[dict, list[str]]:
    if manifest.get("schema_version") not in (1, 2) or manifest.get("kind") != "dl-helper-torch-checkpoint":
        raise ValueError("恢复包类型或 schema_version 不匹配")
    run_id = manifest.get("run_id")
    fingerprint = manifest.get("config_fingerprint")
    revision = manifest.get("source_revision")
    if (not isinstance(run_id, str) or not run_id
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in string.hexdigits for character in fingerprint)
            or (revision is not None and (not isinstance(revision, str) or not revision or any(c.isspace() for c in revision)))):
        raise ValueError("恢复包 run 身份或配置指纹非法")
    if "training_returncode" in manifest and type(manifest["training_returncode"]) is not int:
        raise ValueError("旧恢复包训练退出码非法")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("恢复包文件清单为空或非法")
    names = []
    for name, metadata in files.items():
        path = _safe_member(name)
        if path.parts[0] != "run" or len(path.parts) < 2:
            raise ValueError(f"恢复包路径必须位于 run/ 下：{name}")
        if not isinstance(metadata, dict) or set(metadata) != {"sha256", "size"}:
            raise ValueError(f"恢复包元数据非法：{name}")
        digest = metadata["sha256"]
        size = metadata["size"]
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in string.hexdigits for character in digest)):
            raise ValueError(f"恢复包校验字段非法：{name}")
        names.append(name)
    checkpoint_id = manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or len(_safe_member(checkpoint_id).parts) != 1:
        raise ValueError("恢复包 checkpoint_id 非法")
    prefix = f"run/checkpoints/{checkpoint_id}/"
    required = {"run/config.resolved.yaml", "run/checkpoints/latest.json",
                prefix + "checkpoint-manifest.json"}
    allowed = required | {"run/recovery/terminal-history.json"}
    if any(name not in allowed and not name.startswith((prefix, "run/models/")) for name in names):
        raise ValueError("恢复包含非检查点文件或活动终态")
    if not required <= set(names):
        raise ValueError(f"恢复包缺少必需文件：{sorted(required - set(names))}")
    return manifest, names


def _copy_verified(source: Path, destination: Path, expected: dict, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_stream, destination.open("xb") as target_stream:
        while chunk := source_stream.read(1024 * 1024):
            size += len(chunk)
            if size > expected["size"]:
                raise ValueError(f"恢复包文件超出声明大小：{label}")
            digest.update(chunk)
            target_stream.write(chunk)
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"恢复包文件校验失败：{label}")


def restore_checkpoint_archive(source: Path, destination: Path, expected_run_id: str,
                               expected_source_revision: str, config_path: Path) -> None:
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    config_path = Path(config_path).resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"恢复目标已存在：{destination}")
    manifest, expected_names = _load_resume_manifest(source) if source.is_dir() else (None, None)
    is_zip = source.is_file()
    if not (is_zip or source.is_dir()):
        raise FileNotFoundError(f"恢复来源既非 ZIP 也非目录：{source}")
    if is_zip:
        with ZipFile(source) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise ValueError("恢复 ZIP 含重复路径")
            if "resume-manifest.json" not in names:
                raise ValueError("恢复 ZIP 缺少 resume-manifest.json")
            raw_manifest = archive.read("resume-manifest.json")
            manifest = _read_json_bytes(raw_manifest, "resume-manifest.json")
            manifest, expected_names = _validate_resume_manifest(manifest)
            files = manifest["files"]
            actual_names = set(names) - {"resume-manifest.json"}
            if actual_names != set(expected_names):
                raise ValueError(
                    f"恢复 ZIP 文件集合不符；缺失={sorted(set(expected_names) - actual_names)} "
                    f"多余={sorted(actual_names - set(expected_names))}"
                )
            for info in infos:
                path = _safe_member(info.filename)
                mode = info.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG) or info.is_dir():
                    raise ValueError(f"恢复 ZIP 不允许目录项或符号链接：{info.filename}")
                if path.as_posix() not in {"resume-manifest.json", *expected_names}:
                    raise ValueError(f"恢复 ZIP 包含未声明成员：{info.filename}")
    else:
        actual_names = set()
        for directory, subdirectories, filenames in os.walk(source, followlinks=False):
            base = Path(directory)
            for name in subdirectories:
                if (base / name).is_symlink():
                    raise ValueError(f"恢复目录包含符号链接目录：{base / name}")
            for name in filenames:
                path = base / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"恢复目录包含非普通文件：{path}")
                relative = path.relative_to(source).as_posix()
                if relative != "resume-manifest.json":
                    actual_names.add(relative)
        if actual_names != set(expected_names):
            raise ValueError(
                f"恢复目录文件集合不符；缺失={sorted(set(expected_names) - actual_names)} "
                f"多余={sorted(actual_names - set(expected_names))}"
            )

    manifest, expected_names = _validate_resume_manifest(manifest)
    files = manifest["files"]
    if set(expected_names) != set(files):
        raise ValueError("恢复包文件清单内部不一致")
    if (manifest.get("run_id") != expected_run_id
            or manifest.get("source_revision") != expected_source_revision):
        raise ValueError("恢复包与当前 RUN_ID 或数据版本不匹配")
    expected_config_fingerprint = _config_fingerprint(
        config_path, expected_run_id, expected_source_revision
    )
    if manifest.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("恢复包配置指纹与当前实验配置不匹配")

    _check_space(destination.parent, sum(item["size"] for item in files.values()))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent))
    staged_run = staging_root / "run"
    try:
        staged_run.mkdir()
        if is_zip:
            with ZipFile(source) as archive:
                for name in expected_names:
                    member = _safe_member(name)
                    target = staging_root.joinpath(*member.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(name) as source_stream, target.open("xb") as target_stream:
                        while chunk := source_stream.read(1024 * 1024):
                            size += len(chunk)
                            if size > files[name]["size"]:
                                raise ValueError(f"恢复成员超出声明大小：{name}")
                            digest.update(chunk)
                            target_stream.write(chunk)
                    expected = files[name]
                    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                        raise ValueError(f"恢复包文件校验失败：{name}")
        else:
            for name in expected_names:
                member = _safe_member(name)
                _copy_verified(source.joinpath(*member.parts), staging_root.joinpath(*member.parts),
                               files[name], name)

        latest, checkpoint_dir, checkpoint_manifest = _validate_latest(staged_run, expected_run_id)
        _model_files(staged_run / "models", prefix="")
        if (latest["checkpoint_id"] != manifest.get("checkpoint_id")
                or checkpoint_manifest["config_fingerprint"] != expected_config_fingerprint):
            raise ValueError("恢复包 latest checkpoint 与恢复清单或当前配置不匹配")
        history_path = staged_run / "recovery" / "terminal-history.json"
        if history_path.exists():
            history = _read_json(history_path)
            if history.get("schema_version") != 1 or not isinstance(history.get("entries"), list):
                raise ValueError("terminal-history.json 内容非法")
            for entry in history["entries"]:
                if (not isinstance(entry, dict) or set(entry) != {"name", "content_base64"}
                        or entry["name"] not in ("failure.json", "pause-manifest.json", "run-manifest.json")
                        or not isinstance(entry["content_base64"], str)):
                    raise ValueError("terminal-history.json 记录非法")
                base64.b64decode(entry["content_base64"], validate=True)
        if destination.exists():
            raise FileExistsError(f"恢复目标在恢复期间被创建：{destination}")
        os.replace(staged_run, destination)
    finally:
        shutil.rmtree(staging_root)


def _check_space(parent: Path, required: int) -> None:
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < required:
        raise OSError(f"恢复磁盘空间不足：需要 {required} 字节")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _position(manifest: dict) -> dict:
    position = {key: manifest[key] for key in ("epoch", "global_step", "batch_in_epoch")}
    if any(type(value) is not int or value < 0 for value in position.values()):
        raise ValueError("检查点训练位置非法")
    return position


def _phase(manifest: dict) -> str:
    identifier = manifest["checkpoint_id"]
    if identifier.endswith("-early-stop"):
        return "early-stop"
    if identifier.endswith("-final"):
        return "final"
    return "mid-epoch" if manifest["batch_in_epoch"] else "epoch-boundary"


def _atomic_json(path: Path, value: dict) -> None:
    from .artifacts import atomic_write_text
    atomic_write_text(str(path), _json_bytes(value).decode("utf-8"))


def _inspect_archive(source: Path) -> tuple[dict, dict]:
    """先验证完整 inventory，再读取状态元数据；绝不反序列化模型。"""
    if source.is_symlink():
        raise ValueError(f"恢复来源不能是符号链接：{source}")
    if source.is_file():
        with ZipFile(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("检查点 ZIP 含重复成员")
            for info in infos:
                _safe_member(info.filename)
                if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG):
                    raise ValueError(f"检查点 ZIP 含非普通文件：{info.filename}")
            if archive.getinfo("resume-manifest.json").file_size > 16 * 1024 * 1024:
                raise ValueError("恢复包清单超过16MiB元数据上限")
            outer, expected = _validate_resume_manifest(
                _read_json_bytes(archive.read("resume-manifest.json"), str(source)))
            if set(names) != set(expected) | {"resume-manifest.json"}:
                raise ValueError("检查点 ZIP 文件集合与清单不符")
            for name, metadata in outer["files"].items():
                if archive.getinfo(name).file_size != metadata["size"]:
                    raise ValueError(f"检查点 ZIP 成员大小不符：{name}")
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != metadata["sha256"]:
                    raise ValueError(f"检查点 ZIP SHA256 不符：{name}")
            inner = _read_json_bytes(archive.read(
                f"run/checkpoints/{outer['checkpoint_id']}/checkpoint-manifest.json"), "checkpoint")
            latest = _read_json_bytes(archive.read("run/checkpoints/latest.json"), "latest")
    elif source.is_dir():
        outer, _ = _load_resume_manifest(source)
        _validate_file_inventory(source, outer["files"], excluded={"resume-manifest.json"})
        latest, _, inner = _validate_latest(source / "run", outer["run_id"])
    else:
        raise FileNotFoundError(source)
    if (inner.get("schema_version") != 1 or inner.get("complete") is not True
            or inner.get("backend") != "torch" or latest.get("schema_version") != 1
            or latest.get("checkpoint_id") != outer["checkpoint_id"]
            or latest.get("path") != outer["checkpoint_id"]
            or any(inner.get(key) != outer.get(key)
                   for key in ("run_id", "checkpoint_id", "config_fingerprint"))):
        raise ValueError("恢复包内外清单或 latest 身份不一致")
    position = _position(inner)
    if outer["schema_version"] == 2 and (
            outer.get("backend") != "torch" or outer.get("position") != position
            or outer.get("phase") != _phase(inner)
            or outer.get("data_fingerprint") != inner.get("data_fingerprint")):
        raise ValueError("恢复包阶段、数据或位置不一致")
    return outer, inner


def export_checkpoint_archive(run_dir: Path, checkpoint_dir: Path, output_dir: Path, config) -> Path:
    """在 latest 推进前同步提交 ZIP、独立下载别名及索引。"""
    from .config import config_fingerprint, config_to_dict
    import yaml

    run_dir, checkpoint_dir, output_dir = map(Path, (run_dir, checkpoint_dir, output_dir))
    if any(output_dir.resolve().is_relative_to((run_dir / name).resolve()) for name in ("checkpoints", "models")):
        raise ValueError("检查点 ZIP 不得导出到原始检查点或模型目录")
    inner = _read_json(checkpoint_dir / "checkpoint-manifest.json")
    _validate_file_inventory(checkpoint_dir, inner["files"], excluded={"checkpoint-manifest.json"})
    if (inner.get("complete") is not True or inner.get("backend") != "torch"
            or inner.get("run_id") != (config.run.id or "unknown")
            or inner.get("config_fingerprint") != config_fingerprint(config, resume=True)):
        raise ValueError("导出检查点与当前配置不匹配")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "latest-archive.json"
    if index_path.exists() and _read_json(index_path).get("run_id") != inner["run_id"]:
        raise ValueError("检查点导出目录已属于其他 run")
    identifier = inner["checkpoint_id"]
    _safe_member(identifier)
    target = output_dir / f"checkpoint-{identifier}.zip"
    generated = {
        "run/config.resolved.yaml": yaml.safe_dump(config_to_dict(config), allow_unicode=True,
                                                   sort_keys=False).encode("utf-8"),
        "run/checkpoints/latest.json": _json_bytes(
            {"schema_version": 1, "checkpoint_id": identifier, "path": identifier}),
    }
    history_path = run_dir / "recovery" / "terminal-history.json"
    if history_path.exists():
        _terminal_history(run_dir)  # 在封包前验证历史格式。
        generated["run/recovery/terminal-history.json"] = history_path.read_bytes()
    source_files = {f"run/checkpoints/{identifier}/{name.replace(chr(92), '/')}":
                    checkpoint_dir.joinpath(*name.replace("\\", "/").split("/"))
                    for name in ["checkpoint-manifest.json", *inner["files"]]}
    # 优先使用不可变历史中的辅助权重；旧检查点则封装已恢复到当前 run 的辅助权重。
    model_root = checkpoint_dir / "run-history" / "models"
    if model_root.exists():
        _model_files(model_root, prefix="")  # 已纳入 inner inventory，无需重复压缩一份。
    else:
        source_files.update(_model_files(run_dir / "models", prefix="run/models/"))
    inventory = {name: {"size": path.stat().st_size, "sha256": _sha256_file(path)}
                 for name, path in source_files.items()}
    inventory.update({name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                      for name, raw in generated.items()})
    outer = {"schema_version": 2, "kind": "dl-helper-torch-checkpoint", "backend": "torch",
             "run_id": inner["run_id"], "source_revision": config.run.source_revision,
             "checkpoint_id": identifier, "config_fingerprint": inner["config_fingerprint"],
             "data_fingerprint": inner["data_fingerprint"], "position": _position(inner),
             "phase": _phase(inner), "created_utc": inner["created_utc"], "files": inventory}
    if target.exists():
        previous, previous_inner = _inspect_archive(target)
        if previous_inner != inner or any(previous[key] != outer[key]
                                          for key in ("run_id", "source_revision", "config_fingerprint", "phase", "position")):
            raise ValueError(f"不可变检查点 ZIP 内容冲突：{target}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=output_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for name in sorted(inventory):
                    # 固定成员时间，内容相同的归档不受文件 mtime 影响。
                    from zipfile import ZipInfo
                    info = ZipInfo(name)
                    info.compress_type = ZIP_DEFLATED
                    info._compresslevel = 1
                    with archive.open(info, "w", force_zip64=True) as destination:
                        if name in generated:
                            destination.write(generated[name])
                        else:
                            with source_files[name].open("rb") as source:
                                shutil.copyfileobj(source, destination, length=1024 * 1024)
                archive.writestr("resume-manifest.json", _json_bytes(outer))
            verified, verified_inner = _inspect_archive(temporary)
            if verified != outer or verified_inner != inner:
                raise ValueError("检查点封包后校验失败")
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    descriptor, alias_name = tempfile.mkstemp(prefix=".last-checkpoint-", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    alias = Path(alias_name)
    try:
        with target.open("rb") as source, alias.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(alias, output_dir / "last-checkpoint.zip")
    finally:
        if alias.exists():
            alias.unlink()
    _atomic_json(index_path, {"schema_version": 1, "run_id": inner["run_id"],
                             "checkpoint_id": identifier, "path": target.name,
                             "sha256": _sha256_file(target), "position": _position(inner),
                             "phase": _phase(inner)})
    return target


def _terminal_history(run_dir: Path) -> list[dict]:
    path = run_dir / "recovery" / "terminal-history.json"
    if not path.exists():
        return []  # 此记录是可选历史，缺失意味着未发生终态迁移。
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"终态历史不是普通文件：{path}")
    history = _read_json(path)
    if history.get("schema_version") != 1 or not isinstance(history.get("entries"), list):
        raise ValueError("终态历史格式非法")
    for entry in history["entries"]:
        if (not isinstance(entry, dict) or set(entry) != {"name", "content_base64"}
                or entry["name"] not in ("failure.json", "pause-manifest.json", "run-manifest.json")
                or not isinstance(entry["content_base64"], str)):
            raise ValueError("终态历史记录非法")
        base64.b64decode(entry["content_base64"], validate=True)
    return history["entries"]


def migrate_terminal_history(run_dir: Path, incoming: list[dict] = ()) -> None:
    """有效恢复后将 failure/pause 移入历史；成功 run 不可重跑。"""
    run_dir = Path(run_dir)
    if (run_dir / "run-manifest.json").exists():
        raise ValueError("已成功完成的本地 run 禁止改写")
    history = _terminal_history(run_dir)
    for entry in incoming:
        if entry not in history:
            history.append(entry)
    terminals = [run_dir / name for name in ("failure.json", "pause-manifest.json")
                 if (run_dir / name).exists()]
    if len(terminals) > 1:
        raise ValueError("run 含多个互斥终态")
    for path in terminals:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"终态不是普通文件：{path}")
        entry = {"name": path.name, "content_base64": base64.b64encode(path.read_bytes()).decode("ascii")}
        if entry not in history:
            history.append(entry)
    if history:
        _atomic_json(run_dir / "recovery" / "terminal-history.json", {"schema_version": 1, "entries": history})
    for path in terminals:
        path.unlink()


@dataclass(frozen=True)
class CheckpointSource:
    path: Path
    kind: str
    manifest: dict
    source_revision: str | None
    completed: bool = False

    @property
    def order(self) -> tuple[int, int, int, int]:
        p = _position(self.manifest)
        phase_rank = ("mid-epoch", "epoch-boundary", "early-stop", "final").index(_phase(self.manifest))
        return p["global_step"], p["epoch"], phase_rank, p["batch_in_epoch"]


def _discover(root: Path):
    """仅遍历用户声明的根，不把任意训练/成果 ZIP 当检查点。"""
    if root.is_symlink():
        raise ValueError(f"检查点输入不能是符号链接：{root}")
    if not root.exists():
        return  # 可选固定 Dataset 本次未挂载。
    if root.is_file():
        if root.suffix.lower() != ".zip":
            raise ValueError(f"检查点输入文件必须是 ZIP：{root}")
        yield root, "archive"
        return
    if not root.is_dir():
        raise ValueError(f"检查点输入类型非法：{root}")
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in subdirectories:
            if (base / name).is_symlink():
                raise ValueError(f"检查点输入含符号链接目录：{base / name}")
        if "resume-manifest.json" in filenames:
            yield base, "archive"
            subdirectories[:] = []
            continue
        if "latest-archive.json" in filenames:
            index = _read_json(base / "latest-archive.json")
            member = _safe_member(index["path"])
            if (index.get("schema_version") != 1 or len(member.parts) != 1
                    or _sha256_file(base / index["path"]) != index["sha256"]):
                raise ValueError("最新检查点 ZIP 索引非法或损坏")
            outer, inner = _inspect_archive(base / index["path"])
            if any(index.get(key) != outer.get(key) for key in ("run_id", "checkpoint_id", "position", "phase")):
                raise ValueError("检查点 ZIP 索引与归档不一致")
        if base.name == "checkpoints" and "latest.json" in filenames:
            yield base.parent, "run"
            subdirectories[:] = []
        for name in filenames:
            if name == "last-checkpoint.zip" or (name.startswith("checkpoint-") and name.endswith(".zip")):
                yield base / name, "archive"


def select_checkpoint_source(inputs, run_dir: Path, config) -> CheckpointSource | None:
    from .config import config_fingerprint, load_config_file
    run_dir = Path(run_dir)
    paths = [(run_dir, "local")] if (run_dir / "checkpoints" / "latest.json").exists() else []
    seen_paths = set()
    for root in inputs:
        for path, kind in _discover(Path(root)):
            resolved = path.resolve(strict=True)
            if resolved not in seen_paths and resolved != run_dir.resolve():
                seen_paths.add(resolved)
                paths.append((path, kind))
    candidates = []
    identities = {}
    for path, kind in paths:
        if kind == "archive":
            outer, inner = _inspect_archive(path)
            revision = outer.get("source_revision")
            completed = outer.get("training_returncode") == 0
        else:
            latest = _read_json(path / "checkpoints" / "latest.json")
            identifier = latest["checkpoint_id"]
            _safe_member(identifier)
            inner = _read_json(path / "checkpoints" / identifier / "checkpoint-manifest.json")
            cfg = load_config_file(str(path / "config.resolved.yaml"))
            revision = cfg.run.source_revision
            completed = (path / "run-manifest.json").exists()
        if inner.get("run_id") != config.run.id:
            continue
        if revision != config.run.source_revision or inner.get("config_fingerprint") != config_fingerprint(config, resume=True):
            raise ValueError(f"同 run 检查点配置或 source_revision 不兼容：{path}")
        if kind != "archive":
            _, _, inner = _validate_latest(path, config.run.id)
            if _config_fingerprint(path / "config.resolved.yaml", config.run.id, revision) != inner["config_fingerprint"]:
                raise ValueError("run 配置与检查点不一致")
        identity = inner["checkpoint_id"]
        if identity in identities and identities[identity] != inner:
            raise ValueError(f"相同检查点 ID 的内容或位置冲突：{identity}")
        identities[identity] = inner
        candidate = CheckpointSource(path, kind, inner, revision, completed)
        if completed and inner["epoch"] < config.training.max_epochs and _phase(inner) != "early-stop":
            raise ValueError("旧成功包的检查点早于训练终点，拒绝隐式重复训练")
        candidates.append(candidate)
    return max(candidates, key=lambda item: item.order, default=None)


def restore_checkpoint_source(candidate: CheckpointSource, run_dir: Path, config_path: Path, config) -> None:
    """先在隔离目录完整恢复，安装不可变目录后才推进本地 latest。"""
    from .checkpoint import update_latest
    run_dir = Path(run_dir)
    if candidate.kind == "local":
        migrate_terminal_history(run_dir)
        return
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-import-", dir=run_dir.parent) as temporary:
        staged = Path(temporary) / "run"
        if candidate.kind == "archive":
            restore_checkpoint_archive(candidate.path, staged, config.run.id,
                                       config.run.source_revision, config_path)
        else:
            restore_legacy_run(candidate.path, staged, config.run.id,
                               config.run.source_revision, config_path)
        identifier = candidate.manifest["checkpoint_id"]
        incoming = staged / "checkpoints" / identifier
        target = run_dir / "checkpoints" / identifier
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            current = _read_json(target / "checkpoint-manifest.json")
            _validate_file_inventory(target, current["files"], excluded={"checkpoint-manifest.json"})
            if current != candidate.manifest:
                raise ValueError(f"本地不可变检查点内容冲突：{target}")
        else:
            os.replace(incoming, target)
        models = staged / "models"
        if models.exists():
            # 模型历史在 load_torch_checkpoint 前恢复，项目自定义指标也能读到权重。
            destination = run_dir / "models"
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(models, destination)
        # 历史原子写入成功后才能撤销本地 failure/pause。
        migrate_terminal_history(run_dir, _terminal_history(staged))
        update_latest(str(target.parent), identifier, identifier)


def apply_archive_retention(output_dir: Path, checkpoint_root: Path, keep_last: int | None, run_id: str) -> None:
    """与已完成的原始目录 retention 保持一致，不删除其他 run 或未知文件。"""
    if keep_last is None:
        return
    for path in Path(output_dir).glob("checkpoint-*.zip"):
        with ZipFile(path) as archive:
            outer, _ = _validate_resume_manifest(_read_json_bytes(archive.read("resume-manifest.json"), str(path)))
        if outer["run_id"] == run_id and not (Path(checkpoint_root) / outer["checkpoint_id"]).exists():
            path.unlink()


def _model_files(root: Path, *, prefix: str) -> dict[str, Path]:
    """辅助模型目录可选；存在时要求普通文件，并校验有模型摘要的元数据。"""
    if root.is_symlink():
        raise ValueError(f"辅助模型目录为符号链接：{root}")
    if not root.exists():
        return {}
    if not root.is_dir():
        raise ValueError(f"辅助模型路径不是目录：{root}")
    result = {}
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in subdirectories):
            raise ValueError("辅助模型目录含符号链接")
        for name in filenames:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"辅助模型不是普通文件：{path}")
            result[prefix + path.relative_to(root).as_posix()] = path
            if name.endswith(".json"):
                metadata = _read_json(path)
                if "model_sha256" in metadata:
                    weights = base / "model.safetensors"
                    if not weights.is_file() or weights.is_symlink():
                        raise FileNotFoundError(f"模型元数据缺少对应权重：{weights}")
                    if _sha256_file(weights) != metadata["model_sha256"]:
                        raise ValueError(f"辅助模型 SHA256 不匹配：{weights}")
    return result


def restore_legacy_run(source_run: Path, destination: Path, expected_run_id: str,
                       expected_source_revision: str, config_path: Path) -> None:
    """从旧完整工作目录快照恢复最新检查点。"""
    source_run = Path(source_run)
    if source_run.is_symlink():
        raise ValueError(f"旧 run 目录不能是符号链接：{source_run}")
    source_run = source_run.resolve(strict=True)
    destination = Path(destination).resolve()
    config_path = Path(config_path).resolve(strict=True)
    if not source_run.is_dir() or source_run.is_symlink():
        raise NotADirectoryError(source_run)
    if destination.exists():
        raise FileExistsError(f"恢复目标已存在：{destination}")

    source_config = source_run / "config.resolved.yaml"
    latest_path = source_run / "checkpoints" / "latest.json"
    checkpoint_root = source_run / "checkpoints"
    for path in (source_config, checkpoint_root, latest_path):
        if path.is_symlink():
            raise ValueError(f"旧 run 含符号链接：{path}")
    for path in (source_config, latest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(checkpoint_root)

    source_fingerprint = _config_fingerprint(
        source_config, expected_run_id, expected_source_revision
    )
    current_fingerprint = _config_fingerprint(
        config_path, expected_run_id, expected_source_revision
    )
    if source_fingerprint != current_fingerprint:
        raise ValueError("旧 run 配置与当前 Notebook 生成的配置指纹不匹配")
    latest, source_checkpoint, checkpoint_manifest = _validate_latest(
        source_run, expected_run_id
    )
    if checkpoint_manifest.get("config_fingerprint") != current_fingerprint:
        raise ValueError("旧 checkpoint 配置指纹与当前实验配置不匹配")

    history: list[dict] = []
    previous_history = source_run / "recovery" / "terminal-history.json"
    if previous_history.parent.is_symlink():
        raise ValueError(f"旧恢复目录不能是符号链接：{previous_history.parent}")
    if previous_history.exists():
        if previous_history.is_symlink() or not previous_history.is_file():
            raise ValueError(f"旧终态历史不是普通文件：{previous_history}")
        prior = _read_json(previous_history)
        if prior.get("schema_version") != 1 or not isinstance(prior.get("entries"), list):
            raise ValueError("旧 terminal-history.json 内容非法")
        for entry in prior["entries"]:
            if (not isinstance(entry, dict) or set(entry) != {"name", "content_base64"}
                    or entry["name"] not in ("failure.json", "pause-manifest.json", "run-manifest.json")
                    or not isinstance(entry["content_base64"], str)):
                raise ValueError("旧 terminal-history.json 记录非法")
            base64.b64decode(entry["content_base64"], validate=True)
        history.extend(prior["entries"])

    terminal_names = [name for name in ("failure.json", "pause-manifest.json", "run-manifest.json")
                      if (source_run / name).exists()]
    if len(terminal_names) > 1:
        raise ValueError(f"旧 run 含多个互斥终态文件：{terminal_names}")
    for name in terminal_names:
        path = source_run / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"旧终态文件不是普通文件：{path}")
        history.append({
            "name": name,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.legacy-restore-",
                                         dir=destination.parent))
    staged_run = staging_root / "run"
    try:
        staged_run.mkdir()
        _copy_verified(
            source_config, staged_run / "config.resolved.yaml",
            {"size": source_config.stat().st_size, "sha256": _sha256_file(source_config)},
            "config.resolved.yaml",
        )
        staged_checkpoints = staged_run / "checkpoints"
        staged_checkpoints.mkdir()
        _copy_verified(
            latest_path, staged_checkpoints / "latest.json",
            {"size": latest_path.stat().st_size, "sha256": _sha256_file(latest_path)},
            "checkpoints/latest.json",
        )
        staged_checkpoint = staged_checkpoints / latest["checkpoint_id"]
        staged_checkpoint.mkdir()
        checkpoint_manifest_path = source_checkpoint / "checkpoint-manifest.json"
        if checkpoint_manifest_path.is_symlink() or not checkpoint_manifest_path.is_file():
            raise ValueError(f"旧 checkpoint manifest 不是普通文件：{checkpoint_manifest_path}")
        _copy_verified(
            checkpoint_manifest_path, staged_checkpoint / "checkpoint-manifest.json",
            {"size": checkpoint_manifest_path.stat().st_size,
             "sha256": _sha256_file(checkpoint_manifest_path)},
            "checkpoint-manifest.json",
        )
        for relative, metadata in checkpoint_manifest["files"].items():
            member = _safe_member(relative.replace("\\", "/"))
            _copy_verified(
                source_checkpoint.joinpath(*member.parts),
                staged_checkpoint.joinpath(*member.parts), metadata, relative,
            )
        for relative, path in _model_files(source_run / "models", prefix="models/").items():
            _copy_verified(path, staged_run / relative,
                           {"size": path.stat().st_size, "sha256": _sha256_file(path)}, relative)

        staged_fingerprint = _config_fingerprint(
            staged_run / "config.resolved.yaml", expected_run_id, expected_source_revision
        )
        staged_latest, _, staged_manifest = _validate_latest(staged_run, expected_run_id)
        if (staged_fingerprint != current_fingerprint
                or staged_latest["checkpoint_id"] != latest["checkpoint_id"]
                or staged_manifest.get("config_fingerprint") != current_fingerprint):
            raise ValueError("复制后的旧 checkpoint 与当前 run 身份或配置不匹配")
        if history:
            history_path = staged_run / "recovery" / "terminal-history.json"
            history_path.parent.mkdir(parents=True)
            history_path.write_text(json.dumps(
                {"schema_version": 1, "entries": history}, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")
            ), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"恢复期间目标目录被创建：{destination}")
        os.replace(staged_run, destination)
    finally:
        shutil.rmtree(staging_root)
