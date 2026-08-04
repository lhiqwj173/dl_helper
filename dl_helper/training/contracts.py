"""双后端公共合同：Experiment、DataModule、Task、PredictedBatch 与运行时校验。

本模块只定义稳定公共类型与前置校验，不连接网络、不解析 Secret、
不构造实验，也不导入 transformers 等重库。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Union, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------
# JSON 值类型
# --------------------------------------------------------------------------

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, list["JSONValue"], Mapping[str, "JSONValue"]]


def validate_json_value(value: Any, path: str = "$") -> None:
    """递归校验 value 是否为可规范化的 JSON 值（NaN/Inf 视为非法）。"""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 含非有限浮点数值: {value!r}")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_json_value(item, f"{path}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} 含非字符串键: {key!r}")
            validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} 不是合法 JSON 值: {type(value).__name__}")


# --------------------------------------------------------------------------
# 数据身份与 DataModule
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DataIdentity:
    """非空数据身份；引擎不把路径或文件时间当作可信数据版本。"""

    name: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        for fname in ("name", "version", "fingerprint"):
            value = getattr(self, fname)
            if not isinstance(value, str) or not value:
                raise ValueError(f"DataIdentity.{fname} 必须是非空字符串，得到 {value!r}")


@runtime_checkable
class DataModule(Protocol):
    """通用 PyTorch DataModule 合同。"""

    supports_mid_epoch_resume: bool
    nominal_train_batch_size: int | None

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None: ...

    def train_dataloader(self) -> DataLoader: ...

    def val_dataloader(self) -> DataLoader | None: ...

    def test_dataloader(self) -> DataLoader | None: ...

    def predict_dataloader(self) -> DataLoader | None: ...

    def identity(self) -> DataIdentity: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


def validate_data_module(dm: Any) -> None:
    """显式校验 DataModule 结构，不通过 AttributeError 猜测。"""
    for member in (
        "supports_mid_epoch_resume",
        "nominal_train_batch_size",
        "setup",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "predict_dataloader",
        "identity",
        "state_dict",
        "load_state_dict",
    ):
        if not hasattr(dm, member):
            raise TypeError(f"DataModule 缺少成员 {member!r}: {type(dm).__name__}")
    mid = getattr(dm, "supports_mid_epoch_resume")
    if not isinstance(mid, bool):
        raise TypeError(f"DataModule.supports_mid_epoch_resume 必须为 bool: {mid!r}")
    nominal = getattr(dm, "nominal_train_batch_size")
    if nominal is not None and (not isinstance(nominal, int) or nominal <= 0):
        raise TypeError(f"DataModule.nominal_train_batch_size 必须为正整数或 None: {nominal!r}")
    identity = dm.identity()
    validate_data_identity(identity)


class LoaderDataModule:
    """直接包装用户 DataLoader，支持任意 batch，但不声明中途恢复。"""

    def __init__(
        self,
        identity: DataIdentity,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        predict_dataloader: DataLoader | None = None,
        nominal_train_batch_size: int | None = None,
    ) -> None:
        validate_data_identity(identity)
        if not isinstance(train_dataloader, DataLoader):
            raise TypeError("train_dataloader 必须是 torch DataLoader")
        self._identity = identity
        self._train = train_dataloader
        self._val = val_dataloader
        self._test = test_dataloader
        self._predict = predict_dataloader
        self._nominal = nominal_train_batch_size
        self.supports_mid_epoch_resume = False
        self.nominal_train_batch_size = nominal_train_batch_size

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None:
        return None

    def train_dataloader(self) -> DataLoader:
        return self._train

    def val_dataloader(self) -> DataLoader | None:
        return self._val

    def test_dataloader(self) -> DataLoader | None:
        return self._test

    def predict_dataloader(self) -> DataLoader | None:
        return self._predict

    def identity(self) -> DataIdentity:
        return self._identity

    def split_identity(self, split: str) -> DataIdentity | None:
        """按 split 的实际 dataset 内容生成独立数据指纹。"""
        loader = getattr(self, f"_{split}", None)
        if loader is None:
            return None
        fp = _fingerprint_loader(loader, split)
        return DataIdentity(self._identity.name, self._identity.version, fp)

    def configure_resources(
        self,
        *,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> dict[str, Any]:
        """把平台解析值实际应用到已包装的 DataLoader。"""
        values = {
            "num_workers": self._train.num_workers,
            "pin_memory": self._train.pin_memory,
            "persistent_workers": self._train.persistent_workers,
            "prefetch_factor": getattr(self._train, "prefetch_factor", None),
        }
        requested = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }
        for name, value in requested.items():
            if value is not None:
                values[name] = value
        if not isinstance(values["num_workers"], int) or values["num_workers"] < 0:
            raise ValueError("DataLoader num_workers 必须是非负整数")
        if values["persistent_workers"] and values["num_workers"] == 0:
            raise ValueError("persistent_workers=True 要求 num_workers > 0")
        if values["num_workers"] == 0:
            values["prefetch_factor"] = None
        if self._train.batch_size is None:
            raise ValueError("不支持 batch_size=None 的 LoaderDataModule 资源重建")

        def rebuild(loader: DataLoader | None) -> DataLoader | None:
            if loader is None:
                return None
            if loader.batch_size is None:
                raise ValueError("不支持 batch_size=None 的 LoaderDataModule 资源重建")
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                sampler=loader.sampler,
                num_workers=values["num_workers"],
                collate_fn=loader.collate_fn,
                pin_memory=values["pin_memory"],
                drop_last=loader.drop_last,
                timeout=loader.timeout,
                worker_init_fn=loader.worker_init_fn,
                multiprocessing_context=loader.multiprocessing_context,
                generator=loader.generator,
                prefetch_factor=values["prefetch_factor"],
                persistent_workers=values["persistent_workers"],
            )

        self._train = rebuild(self._train)
        self._val = rebuild(self._val)
        self._test = rebuild(self._test)
        self._predict = rebuild(self._predict)
        return dict(values)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("LoaderDataModule 不支持中途恢复，state 必须为空")


class ResumableMapDataModule:
    """由 dataset/collate 工厂与 DataLoader 参数构造，保存 epoch/已消费批次数与 sampler 状态。

    严格确定性模式下要求随机增强由样本键、epoch 和 seed 决定，不依赖未保存的 worker 全局 RNG。
    """

    def __init__(
        self,
        identity: DataIdentity,
        dataset_factory: Callable[[], torch.utils.data.Dataset],
        collate_fn: Callable[[list[Any]], Any],
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        val_dataset_factory: Callable[[], torch.utils.data.Dataset] | None = None,
        test_dataset_factory: Callable[[], torch.utils.data.Dataset] | None = None,
        predict_dataset_factory: Callable[[], torch.utils.data.Dataset] | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        predict_batch_size: int | None = None,
    ) -> None:
        validate_data_identity(identity)
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        self._identity = identity
        self._dataset_factory = dataset_factory
        self._collate_fn = collate_fn
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._prefetch_factor = prefetch_factor
        self._persistent_workers = persistent_workers
        self._val_dataset_factory = val_dataset_factory
        self._test_dataset_factory = test_dataset_factory
        self._predict_dataset_factory = predict_dataset_factory
        self._val_batch_size = val_batch_size or batch_size
        self._test_batch_size = test_batch_size or batch_size
        self._predict_batch_size = predict_batch_size or batch_size
        self.supports_mid_epoch_resume = True
        self.nominal_train_batch_size = batch_size
        self._dataset = dataset_factory()
        self._val_dataset = None if val_dataset_factory is None else val_dataset_factory()
        self._test_dataset = None if test_dataset_factory is None else test_dataset_factory()
        self._predict_dataset = None if predict_dataset_factory is None else predict_dataset_factory()
        self._epoch = 0
        self._consumed_batches = 0
        self._generator: torch.Generator | None = None
        self._worker_seeds: list[int] | None = None
        self._epoch_rng_state: torch.ByteTensor | None = None

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None:
        return None

    def configure_resources(
        self,
        *,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> dict[str, Any]:
        """OSR-006：应用平台解析的 DataLoader 资源（必须在首次创建 loader 前调用）。

        平台解析值必须原样应用；不得按数据集大小静默改写资源合同。
        """
        if num_workers is not None:
            if not isinstance(num_workers, int) or num_workers < 0:
                raise ValueError("num_workers 必须是非负整数")
            self._num_workers = num_workers
        if pin_memory is not None:
            self._pin_memory = pin_memory
        if persistent_workers is not None:
            if persistent_workers and self._num_workers == 0:
                raise ValueError("persistent_workers=True 要求 num_workers > 0")
            self._persistent_workers = persistent_workers
        if prefetch_factor is not None:
            if self._num_workers == 0:
                self._prefetch_factor = None
            else:
                if not isinstance(prefetch_factor, int) or prefetch_factor <= 0:
                    raise ValueError("prefetch_factor 必须是正整数")
                self._prefetch_factor = prefetch_factor
        elif self._num_workers == 0:
            self._prefetch_factor = None
        return {
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "prefetch_factor": self._prefetch_factor,
        }

    def applied_loader_resources(self) -> dict[str, Any]:
        """当前 loader 资源的实际应用值（供 environment/manifest 记录）。"""
        return {
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "prefetch_factor": self._prefetch_factor,
        }

    def _dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int) -> DataLoader:
        gen = None
        if self._shuffle:
            # OSR-004：每 epoch 由 epoch 确定性种子派生 shuffle；恢复后引擎按
            # batch_in_epoch 跳过可精确复现同一批样本，不依赖已消费的 generator
            # 状态定位 loader（避免状态化 sampler 双重跳过）。
            gen = torch.Generator()
            gen.manual_seed(self._epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self._shuffle,
            sampler=None,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
            generator=gen,
            drop_last=False,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._dataset, self._batch_size)

    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is None:
            return None
        return self._dataloader(self._val_dataset, self._val_batch_size)

    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is None:
            return None
        return self._dataloader(self._test_dataset, self._test_batch_size)

    def predict_dataloader(self) -> DataLoader | None:
        if self._predict_dataset is None:
            return None
        return self._dataloader(self._predict_dataset, self._predict_batch_size)

    def identity(self) -> DataIdentity:
        return self._identity

    def split_identity(self, split: str) -> DataIdentity | None:
        """OSR-005：per-split 数据指纹（identity + split 名 + 样本数派生，不同 split 不同）。"""
        dataset = {"train": self._dataset, "val": self._val_dataset,
                   "test": self._test_dataset, "predict": self._predict_dataset}.get(split)
        if dataset is None:
            return None
        fp = _fingerprint_dataset(dataset, split)
        return DataIdentity(self._identity.name, self._identity.version, fp)

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "epoch": self._epoch,
            "consumed_batches": self._consumed_batches,
        }
        if self._generator is not None:
            state["generator_state"] = self._generator.get_state().numpy().copy()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "epoch" not in state or "consumed_batches" not in state:
            raise ValueError("ResumableMapDataModule state 缺少 epoch/consumed_batches")
        self._epoch = int(state["epoch"])
        self._consumed_batches = int(state["consumed_batches"])
        gs = state.get("generator_state")
        if gs is not None:
            self._generator = torch.Generator()
            self._generator.set_state(torch.as_tensor(gs, dtype=torch.uint8))

    def advance_batch(self) -> None:
        self._consumed_batches += 1

    def advance_epoch(self) -> None:
        self._epoch += 1
        self._consumed_batches = 0


def validate_data_identity(identity: Any) -> None:
    if not isinstance(identity, DataIdentity):
        raise TypeError(f"identity 必须是 DataIdentity: {type(identity).__name__}")
    identity.__post_init__()


def contract_splits(datamodule: Any, identity: DataIdentity) -> dict[str, dict[str, str]]:
    """按实际 split 内容生成合同；缺少可复核声明时直接失败。"""
    splits: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test", "predict"):
        if split == "train" and hasattr(datamodule, "incremental_train_data"):
            source = datamodule.incremental_train_data()
            if source is None:
                raise ValueError("incremental_train_data() 不得返回 None")
            fp = _fingerprint_incremental_source(source, split)
            splits[split] = {"fingerprint": fp, "name": identity.name, "version": identity.version}
            continue
        if not _split_available(datamodule, split):
            continue
        split_identity = getattr(datamodule, "split_identity", None)
        if callable(split_identity):
            split_data_identity = split_identity(split)
            if split_data_identity is None:
                raise ValueError(f"DataModule.split_identity({split!r}) 未声明")
            validate_data_identity(split_data_identity)
            fp = split_data_identity.fingerprint
        else:
            fp = _fingerprint_split(datamodule, split)
        splits[split] = {"fingerprint": fp, "name": identity.name, "version": identity.version}
    if "train" not in splits:
        raise ValueError("evaluation contract 缺少 train split 的真实指纹")
    return splits


def _split_available(datamodule: Any, split: str) -> bool:
    if hasattr(datamodule, f"{split}_dataloader"):
        return getattr(datamodule, f"{split}_dataloader")() is not None
    if hasattr(datamodule, "evaluation_batches"):
        return any(True for _ in datamodule.evaluation_batches(split))
    return False


def _hash_value(hasher: Any, value: Any) -> None:
    """稳定编码 tensor/ndarray/容器，避免只依据样本数构造 fingerprint。"""
    import torch

    if isinstance(value, torch.Tensor):
        _hash_value(hasher, value.detach().cpu().contiguous().numpy())
        return
    if isinstance(value, np.ndarray):
        hasher.update(b"ndarray:")
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(json.dumps(list(value.shape)).encode("utf-8"))
        hasher.update(value.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping:")
        for key in sorted(value):
            _hash_value(hasher, key)
            _hash_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _hash_value(hasher, item)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        hasher.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return
    hasher.update(repr(value).encode("utf-8"))


def _fingerprint_dataset(dataset: Any, split: str) -> str:
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise ValueError(f"{split} dataset 必须提供可复核的 __len__/__getitem__")
    hasher = hashlib.sha256(f"split:{split}".encode("utf-8"))
    count = len(dataset)
    hasher.update(str(count).encode("utf-8"))
    for index in range(count):
        _hash_value(hasher, dataset[index])
    return hasher.hexdigest()


def _fingerprint_loader(loader: DataLoader, split: str) -> str:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError(f"{split} DataLoader 缺少 dataset，无法构造真实指纹")
    return _fingerprint_dataset(dataset, split)


def _fingerprint_split(datamodule: Any, split: str) -> str:
    return _fingerprint_batches(datamodule.evaluation_batches(split), split)


def _fingerprint_incremental_source(source: Any, split: str) -> str:
    """指纹化增量训练源，并保证合同生成不永久推进 source 状态。"""
    for member in ("iter_epoch", "state_dict", "load_state_dict"):
        if not callable(getattr(source, member, None)):
            raise TypeError(f"增量训练源缺少可恢复成员 {member!r}")
    state = copy.deepcopy(source.state_dict())
    try:
        return _fingerprint_batches(source.iter_epoch(0), split)
    finally:
        source.load_state_dict(state)


def _fingerprint_batches(batches: Iterable[Any], split: str) -> str:
    hasher = hashlib.sha256(f"split:{split}".encode("utf-8"))
    count = 0
    for batch in batches:
        sample_count = getattr(batch, "sample_count", None)
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(f"{split} batch.sample_count 必须为正整数: {sample_count!r}")
        _hash_value(hasher, getattr(batch, "features", None))
        _hash_value(hasher, getattr(batch, "targets", None))
        _hash_value(hasher, getattr(batch, "sample_weight", None))
        _hash_value(hasher, getattr(batch, "sample_ids", None))
        count += sample_count
    if count <= 0:
        raise ValueError(f"{split} split 为空，无法构造评价合同")
    hasher.update(str(count).encode("utf-8"))
    return hasher.hexdigest()


def contract_label_schema(task: Any) -> dict[str, Any] | None:
    """完整 label/target schema（含类别/标签顺序与多标签 threshold）。"""
    num_classes = getattr(task, "num_classes", None)
    if num_classes is None and getattr(task, "classes", None) is not None:
        num_classes = len(getattr(task, "classes"))
    num_labels = getattr(task, "num_labels", None)
    num_targets = getattr(task, "num_targets", None)
    if num_classes is not None:
        classes = getattr(task, "classes", None)
        if classes is None:
            raise ValueError("分类 Task 必须声明有序 classes")
        return {
            "kind": "classification",
            "num_classes": int(num_classes),
            "classes": [_json_scalar(v) for v in list(classes)],
        }
    if num_labels is not None:
        schema: dict[str, Any] = {"kind": "multilabel", "num_labels": int(num_labels)}
        thr = getattr(task, "threshold", None)
        if thr is not None:
            if isinstance(thr, (list, tuple, np.ndarray)):
                schema["threshold"] = [float(t) for t in thr]
            else:
                schema["threshold"] = float(thr)
        labels = getattr(task, "label_names", None)
        if labels is None or len(labels) != int(num_labels):
            raise ValueError("多标签 Task 必须声明与 num_labels 等长的 label_names")
        schema["labels"] = [_json_scalar(v) for v in labels]
        schema["label_names"] = list(schema["labels"])
        return schema
    if num_targets is not None:
        names = getattr(task, "target_names", None)
        if names is None or len(names) != int(num_targets):
            raise ValueError("回归 Task 必须声明与 num_targets 等长的 target_names")
        return {
            "kind": "regression",
            "num_targets": int(num_targets),
            "target_names": [_json_scalar(v) for v in names],
        }
    return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"合同标签值不是 JSON 标量: {value!r}")


def contract_metric_definitions(task: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": definition.name,
            "direction": definition.direction,
            "formula_id": definition.formula_id,
            "formula_version": definition.formula_version,
            "averaging": definition.averaging,
            "sample_weight_policy": definition.sample_weight_policy,
            "zero_division": definition.zero_division,
            "exact": definition.exact,
            "evaluation_scope": definition.evaluation_scope,
            "parameters": dict(definition.parameters),
            "implementation": definition.implementation,
        }
        for name, definition in task.metric_definitions.items()
    }


def build_evaluation_contract(
    datamodule: Any,
    task: Any,
    backend: str,
    model_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = datamodule.identity()
    validate_data_identity(identity)
    label_schema = contract_label_schema(task)
    if label_schema is None:
        raise ValueError("Task 未声明完整 label/target schema")
    contract: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "data_identity": {
            "name": identity.name,
            "version": identity.version,
            "fingerprint": identity.fingerprint,
        },
        "splits": contract_splits(datamodule, identity),
        "label_schema": label_schema,
        "task_name": getattr(task, "name", None),
        "metric_definitions": contract_metric_definitions(task),
    }
    if model_signature is not None:
        contract["model_signature"] = dict(model_signature)
    return contract


# --------------------------------------------------------------------------
# sklearn 数据合同
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorBatch:
    features: Any
    targets: Any
    sample_count: int
    sample_weight: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None


@runtime_checkable
class IncrementalBatchSource(Protocol):
    classes: np.ndarray | None
    nominal_batch_size: int | None
    supports_mid_fit_resume: bool

    def iter_epoch(self, epoch: int) -> Iterable[EstimatorBatch]: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class SklearnEvaluationDataModule(Protocol):
    def setup(self, stage: Literal["fit", "test", "predict"]) -> None: ...

    def evaluation_batches(self, stage: Literal["train", "val", "test", "predict"]) -> Iterable[EstimatorBatch]: ...

    def identity(self) -> DataIdentity: ...


@runtime_checkable
class SklearnBatchDataModule(SklearnEvaluationDataModule, Protocol):
    def full_train_data(self) -> EstimatorBatch: ...


@runtime_checkable
class SklearnIncrementalDataModule(SklearnEvaluationDataModule, Protocol):
    def incremental_train_data(self) -> IncrementalBatchSource: ...


def validate_sklearn_batch_datamodule(dm: Any) -> None:
    for member in ("setup", "evaluation_batches", "identity", "full_train_data"):
        if not hasattr(dm, member):
            raise TypeError(f"SklearnBatchDataModule 缺少成员 {member!r}: {type(dm).__name__}")


def validate_sklearn_incremental_datamodule(dm: Any) -> None:
    for member in ("setup", "evaluation_batches", "identity", "incremental_train_data"):
        if not hasattr(dm, member):
            raise TypeError(f"SklearnIncrementalDataModule 缺少成员 {member!r}: {type(dm).__name__}")


# --------------------------------------------------------------------------
# PreparedBatch 与 LossResult
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedBatch:
    inputs: Any
    targets: Any
    sample_count: int
    sample_weight: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_prepared_batch(self)


@dataclass(frozen=True)
class LossResult:
    numerator: torch.Tensor
    denominator: torch.Tensor | int | float

    def __post_init__(self) -> None:
        validate_loss_result(self)


def validate_prepared_batch(prepared: PreparedBatch) -> None:
    if not isinstance(prepared.sample_count, int) or prepared.sample_count <= 0:
        raise ValueError(f"PreparedBatch.sample_count 必须为正整数: {prepared.sample_count!r}")
    if prepared.sample_weight is not None:
        w = prepared.sample_weight
        if not isinstance(w, torch.Tensor) or w.dim() != 1:
            raise ValueError("PreparedBatch.sample_weight 必须是一维 Tensor")
        if w.shape[0] != prepared.sample_count:
            raise ValueError(
                f"PreparedBatch.sample_weight 长度 {w.shape[0]} 与 sample_count {prepared.sample_count} 不一致"
            )
        if not torch.isfinite(w).all():
            raise ValueError("PreparedBatch.sample_weight 含非有限值")
        if (w < 0).any():
            raise ValueError("PreparedBatch.sample_weight 含负值")
        if float(w.sum()) <= 0:
            raise ValueError("PreparedBatch.sample_weight 批内权重和必须为正")


def validate_loss_result(loss: LossResult) -> None:
    num = loss.numerator
    if not isinstance(num, torch.Tensor) or num.dim() != 0:
        raise ValueError("LossResult.numerator 必须是标量 Tensor")
    if not torch.isfinite(num):
        raise ValueError("LossResult.numerator 必须有限")
    # 训练上下文（grad 开启）要求可微；eval（no_grad）只取数值
    if torch.is_grad_enabled() and not num.requires_grad:
        raise ValueError("LossResult.numerator 必须要求梯度（可微）")
    den = loss.denominator
    if isinstance(den, torch.Tensor):
        if den.dim() != 0:
            raise ValueError("LossResult.denominator 必须是标量")
        if den.requires_grad:
            raise ValueError("LossResult.denominator 不得带梯度")
        dval = float(den)
    elif isinstance(den, (int, float)):
        dval = float(den)
    else:
        raise TypeError(f"LossResult.denominator 类型非法: {type(den).__name__}")
    if not math.isfinite(dval) or dval <= 0:
        raise ValueError(f"LossResult.denominator 必须是有限正数: {dval!r}")


# --------------------------------------------------------------------------
# PredictedBatch 与 Task
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictedBatch:
    targets: Any
    predictions: Any
    sample_count: int
    scores: Any | None = None
    sample_weight: Any | None = None
    sample_ids: Any | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_predicted_batch(self)


def validate_predicted_batch(predicted: PredictedBatch) -> None:
    n = predicted.sample_count
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"PredictedBatch.sample_count 必须为正整数: {n!r}")
    targets = predicted.targets
    if isinstance(targets, np.ndarray) and targets.ndim > 0 and targets.shape[0] != n:
        raise ValueError(
            f"PredictedBatch.targets 样本维 {targets.shape[0]} 与 sample_count {n} 不一致"
        )
    for fname in ("predictions", "scores", "sample_weight", "sample_ids"):
        value = getattr(predicted, fname)
        if value is None:
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(f"PredictedBatch.{fname} 必须是 np.ndarray: {type(value).__name__}")
        if value.ndim > 0 and value.shape[0] != n:
            raise ValueError(
                f"PredictedBatch.{fname} 样本维 {value.shape[0]} 与 sample_count {n} 不一致"
            )
        if value.dtype.kind in "fc":
            if not np.isfinite(value).all():
                raise ValueError(f"PredictedBatch.{fname} 含非有限值")
        if fname == "sample_weight":
            if value.ndim != 1:
                raise ValueError("PredictedBatch.sample_weight 必须是一维")
            if value.dtype.kind not in "fc":
                raise TypeError("PredictedBatch.sample_weight 必须是浮点 dtype")
            if np.any(value < 0):
                raise ValueError("PredictedBatch.sample_weight 含负值")
            if float(np.sum(value)) <= 0:
                raise ValueError("PredictedBatch.sample_weight 权重和必须为正")


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: Literal["min", "max"]
    formula_id: str
    formula_version: int
    averaging: Literal["none", "micro", "macro", "weighted", "uniform_average", "variance_weighted"]
    sample_weight_policy: Literal["supported", "required", "forbidden"]
    zero_division: Literal["zero", "one", "error", "not_applicable"]
    exact: bool
    evaluation_scope: Literal["full", "sampled"]
    parameters: Mapping[str, JSONValue]
    implementation: Literal["builtin_verified", "custom"]

    def __post_init__(self) -> None:
        validate_metric_definition(self)


def validate_metric_definition(defn: MetricDefinition) -> None:
    if not isinstance(defn.name, str) or not defn.name:
        raise ValueError("MetricDefinition.name 必须是非空字符串")
    if defn.direction not in ("min", "max"):
        raise ValueError(f"MetricDefinition.direction 非法: {defn.direction!r}")
    if not isinstance(defn.formula_id, str) or not defn.formula_id:
        raise ValueError("MetricDefinition.formula_id 必须是非空字符串")
    if not isinstance(defn.formula_version, int) or defn.formula_version <= 0:
        raise ValueError(f"MetricDefinition.formula_version 必须为正整数: {defn.formula_version!r}")
    allowed_averaging = (
        "none", "micro", "macro", "weighted", "uniform_average", "variance_weighted",
    )
    if defn.averaging not in allowed_averaging:
        raise ValueError(f"MetricDefinition.averaging 非法: {defn.averaging!r}")
    if defn.sample_weight_policy not in ("supported", "required", "forbidden"):
        raise ValueError(f"MetricDefinition.sample_weight_policy 非法: {defn.sample_weight_policy!r}")
    if defn.zero_division not in ("zero", "one", "error", "not_applicable"):
        raise ValueError(f"MetricDefinition.zero_division 非法: {defn.zero_division!r}")
    if not isinstance(defn.exact, bool):
        raise ValueError("MetricDefinition.exact 必须为 bool")
    if defn.evaluation_scope not in ("full", "sampled"):
        raise ValueError(f"MetricDefinition.evaluation_scope 非法: {defn.evaluation_scope!r}")
    if defn.implementation not in ("builtin_verified", "custom"):
        raise ValueError(f"MetricDefinition.implementation 非法: {defn.implementation!r}")
    validate_json_value(dict(defn.parameters), f"MetricDefinition({defn.name}).parameters")
    if defn.evaluation_scope == "sampled" and defn.exact:
        raise ValueError("sampled 指标不得标记 exact=true")


@runtime_checkable
class MetricState(Protocol):
    def reset(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum", "min", "max", "merge_weighted_moments"]]]: ...

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None: ...

    def compute(self) -> Mapping[str, float]: ...


@runtime_checkable
class EvaluationTask(Protocol):
    name: str
    metric_definitions: Mapping[str, MetricDefinition]

    def metric_state(self, stage: str) -> MetricState: ...

    def update_metrics(self, state: MetricState, predicted: PredictedBatch) -> None: ...

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]: ...

    def report_kind(self) -> str: ...


@runtime_checkable
class TorchTask(EvaluationTask, Protocol):
    def prepare_batch(self, batch: Any, stage: str) -> PreparedBatch: ...

    def forward(self, model: torch.nn.Module, prepared: PreparedBatch) -> Any: ...

    def loss(self, outputs: Any, prepared: PreparedBatch) -> LossResult: ...

    def to_predicted_batch(self, outputs: Any, prepared: PreparedBatch) -> PredictedBatch: ...


@runtime_checkable
class SklearnTask(EvaluationTask, Protocol):
    estimator_kind: Literal["classifier", "regressor"]
    classes: np.ndarray | None
    required_prediction: Literal["predict", "decision_function", "predict_proba"]

    def predict_batch(self, estimator: Any, batch: EstimatorBatch) -> PredictedBatch: ...


def validate_torch_task(task: Any) -> None:
    for member in ("name", "metric_definitions", "metric_state", "update_metrics",
                   "prediction_arrays", "report_kind", "prepare_batch", "forward",
                   "loss", "to_predicted_batch"):
        if not hasattr(task, member):
            raise TypeError(f"TorchTask 缺少成员 {member!r}: {type(task).__name__}")
    validate_task_common(task)


def validate_sklearn_task(task: Any) -> None:
    for member in ("name", "metric_definitions", "metric_state", "update_metrics",
                   "prediction_arrays", "report_kind", "estimator_kind", "classes",
                   "required_prediction", "predict_batch"):
        if not hasattr(task, member):
            raise TypeError(f"SklearnTask 缺少成员 {member!r}: {type(task).__name__}")
    if getattr(task, "estimator_kind") not in ("classifier", "regressor"):
        raise ValueError(f"SklearnTask.estimator_kind 非法: {getattr(task, 'estimator_kind')!r}")
    if getattr(task, "required_prediction") not in ("predict", "decision_function", "predict_proba"):
        raise ValueError(f"SklearnTask.required_prediction 非法: {getattr(task, 'required_prediction')!r}")
    validate_task_common(task)


def validate_task_common(task: Any) -> None:
    name = getattr(task, "name")
    if not isinstance(name, str) or not name:
        raise ValueError("Task.name 必须是非空字符串")
    defs = getattr(task, "metric_definitions")
    if not isinstance(defs, Mapping) or not defs:
        raise ValueError("Task.metric_definitions 必须是非空 mapping")
    for key, defn in defs.items():
        if not isinstance(defn, MetricDefinition):
            raise TypeError(f"metric_definitions[{key!r}] 必须是 MetricDefinition")
        if defn.name != key:
            raise ValueError(f"metric_definitions 键 {key!r} 与定义名 {defn.name!r} 不一致")
        if defn.evaluation_scope == "full" and defn.exact is False:
            raise ValueError(f"full 指标 {key!r} 必须 exact=true")


# --------------------------------------------------------------------------
# SchedulerBinding
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerBinding:
    scheduler: Any
    interval: Literal["optimizer_step", "epoch", "validation_metric"]
    monitor: str | None

    def __post_init__(self) -> None:
        if self.interval not in ("optimizer_step", "epoch", "validation_metric"):
            raise ValueError(f"SchedulerBinding.interval 非法: {self.interval!r}")
        if self.interval == "validation_metric" and not self.monitor:
            raise ValueError("validation_metric scheduler 必须配置 monitor")
        if self.interval != "validation_metric" and self.monitor is not None:
            raise ValueError("非 validation_metric scheduler 的 monitor 必须为 None")


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TorchExperiment:
    name: str
    backend: Literal["torch"]
    model_factory: Callable[[], torch.nn.Module]
    datamodule_factory: Callable[[], DataModule]
    task_factory: Callable[[], TorchTask]
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
    scheduler_factory: Callable[[torch.optim.Optimizer], SchedulerBinding | None]
    model_config: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SklearnExperiment:
    name: str
    backend: Literal["sklearn"]
    estimator_factory: Callable[[], Any]
    datamodule_factory: Callable[[], Any]
    task_factory: Callable[[], SklearnTask]
    model_config: Mapping[str, JSONValue] = field(default_factory=dict)


Experiment = Union[TorchExperiment, SklearnExperiment]


def validate_experiment(experiment: Any) -> None:
    if not isinstance(experiment, (TorchExperiment, SklearnExperiment)):
        raise TypeError(
            f"build_experiment 必须返回 TorchExperiment 或 SklearnExperiment，得到 {type(experiment).__name__}"
        )
    if not isinstance(experiment.name, str) or not experiment.name:
        raise ValueError("Experiment.name 必须是非空字符串")
    validate_json_value(dict(experiment.model_config), f"Experiment({experiment.name}).model_config")
    if isinstance(experiment, TorchExperiment):
        for fname in ("model_factory", "datamodule_factory", "task_factory",
                      "optimizer_factory", "scheduler_factory"):
            if not callable(getattr(experiment, fname)):
                raise TypeError(f"TorchExperiment.{fname} 必须可调用")
    else:
        for fname in ("estimator_factory", "datamodule_factory", "task_factory"):
            if not callable(getattr(experiment, fname)):
                raise TypeError(f"SklearnExperiment.{fname} 必须可调用")


def validate_backend_match(experiment: Experiment, backend_type: str) -> None:
    if isinstance(experiment, TorchExperiment) and backend_type != "torch":
        raise ValueError(f"TorchExperiment 与 backend.type={backend_type!r} 不匹配")
    if isinstance(experiment, SklearnExperiment) and backend_type != "sklearn":
        raise ValueError(f"SklearnExperiment 与 backend.type={backend_type!r} 不匹配")
