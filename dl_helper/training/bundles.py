"""与远端服务无关的本地成果归档；原始运行清单保持不变。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import ArtifactError, atomic_write_text, existing_terminal, read_json, sha256_file

EXCLUDED_DIRECTORIES = ("checkpoints", "checkpoint-archives")
EXCLUDED_FILES = ("services/bundle-published.json",)
BUNDLE_MANIFEST = "bundle-manifest.json"


def default_bundle_path(run_dir: str | Path) -> Path:
    """在运行目录旁使用独立文件名，防止相互覆盖或与 run ID 目录冲突。"""
    root = Path(run_dir).resolve(strict=True)
    return root.parent / f"{root.name}.run-bundle.zip"


def _excluded(name: str, exclude_prefixes: tuple[str, ...]) -> bool:
    return name in EXCLUDED_FILES or any(
        name == prefix or name.startswith(prefix.rstrip("/") + "/") for prefix in exclude_prefixes
    )


def _members(root: Path, exclude_prefixes: tuple[str, ...]) -> dict[str, Path]:
    def walk_error(error: OSError) -> None:
        raise error

    files: dict[str, Path] = {}
    for current, directories, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        for name in sorted(directories + filenames):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if name.startswith(".") or _excluded(relative, exclude_prefixes):
                continue
            if path.is_symlink() or (os.name == "nt" and path.lstat().st_file_attributes
                                     & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ArtifactError(f"归档成员为链接: {relative}")
            if "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/")):
                raise ArtifactError(f"归档成员路径非法: {relative}")
            if path.is_file():
                if relative == BUNDLE_MANIFEST:
                    raise ArtifactError(f"源目录包含保留的归档清单名称: {relative}")
                if name.endswith(".tmp") or ".tmp." in name:
                    raise ArtifactError(f"存在未完成的成果文件: {relative}")
                files[relative] = path
            elif not path.is_dir():
                raise ArtifactError(f"归档成员不是普通文件或目录: {relative}")
        directories[:] = sorted(name for name in directories
                                if not name.startswith(".") and not _excluded(
                                    (Path(current) / name).relative_to(root).as_posix(), exclude_prefixes))
    return dict(sorted(files.items()))


def write_directory_bundle(
    source_dir: str | Path, output: str | Path, *,
    exclude_prefixes: tuple[str, ...] = EXCLUDED_DIRECTORIES,
    expected_artifacts: dict | None = None,
) -> Path:
    """流式写入确定性 ZIP，逐文件回读校验后发布；适用于 run/sweep 与远端上传。"""
    root = Path(source_dir).resolve(strict=True)
    target = Path(output).resolve()
    if not root.is_dir() or target.is_relative_to(root):
        raise ArtifactError("成果源必须是目录，ZIP 输出必须位于源目录之外")
    files = _members(root, exclude_prefixes)
    if not files:
        raise ArtifactError("成果目录为空")
    inventory = {name: {"size": path.stat().st_size, "sha256": sha256_file(str(path))}
                 for name, path in files.items()}
    if expected_artifacts is not None:
        for name, expected in expected_artifacts.items():
            name = name.replace("\\", "/")
            if _excluded(name, exclude_prefixes):
                continue
            if name not in inventory or inventory[name] != expected:
                raise ArtifactError(f"运行清单与成果文件不一致: {name}")
    manifest = {"schema_version": 1, "kind": "result_bundle",
                "excluded_prefixes": list(exclude_prefixes),
                "excluded_files": list(EXCLUDED_FILES), "files": inventory}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    # 随机临时文件避免不同运行/进程共用固定 .tmp 名称。
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted([*files, BUNDLE_MANIFEST]):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = ZIP_DEFLATED
                info._compresslevel = 6
                info.external_attr = 0o100644 << 16
                with archive.open(info, "w", force_zip64=True) as destination:
                    if name == BUNDLE_MANIFEST:
                        destination.write(manifest_bytes)
                    else:
                        with files[name].open("rb") as source:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
        with ZipFile(temporary) as archive:
            if set(archive.namelist()) != set(files) | {BUNDLE_MANIFEST}:
                raise ArtifactError("ZIP 成员列表不一致")
            for name, expected in inventory.items():
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if (archive.getinfo(name).file_size != expected["size"]
                        or digest.hexdigest() != expected["sha256"]):
                    raise ArtifactError(f"ZIP 回读校验失败: {name}")
            if archive.read(BUNDLE_MANIFEST) != manifest_bytes:
                raise ArtifactError("ZIP 归档清单校验失败")
        archive_sha256 = sha256_file(str(temporary))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
        atomic_write_text(str(target.with_suffix(target.suffix + ".sha256")),
                          f"{archive_sha256}  {target.name}\n")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def export_run_bundle(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """导出成功运行；可在业务后处理之后重用，打包失败直接抛出异常。"""
    root = Path(run_dir).resolve(strict=True)
    if existing_terminal(str(root)) != "run-manifest.json":
        raise ArtifactError("只有成功运行可以导出最终成果包；暂停时使用恢复检查点 ZIP")
    manifest = read_json(str(root / "run-manifest.json"))
    if manifest["status"] != "succeeded" or not manifest["run_id"]:
        raise ArtifactError("运行清单不是有效的成功终态")
    return write_directory_bundle(root, output if output is not None else default_bundle_path(root),
                                  expected_artifacts=manifest["artifacts"])
