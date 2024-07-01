import torch
from accelerate import Accelerator
from torch.utils.data import BatchSampler, DataLoader, IterableDataset
from accelerate.state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from accelerate.data_loader import _PYTORCH_DATALOADER_KWARGS, DataLoaderDispatcher, DataLoaderShard, MpDeviceLoaderWrapper


if is_torch_xla_available():
    # import torch_xla.distributed.parallel_loader as xpl
    from dl_helper.acc.parallel_loader import MpDeviceLoader

    # class MpDeviceLoaderWrapper(xpl.MpDeviceLoader):
    class MpDeviceLoaderWrapper2(MpDeviceLoader):
        """
        Wrapper for the xpl.MpDeviceLoader class that knows the total batch size.

        XLA preloading threads will all call DataLoaderShard's __iter__(). Remove rng_types from DataLoaderShard to
        prevent it from using the XLA device in the preloading threads, and synchronize the RNG once from the main
        thread only.

        **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
        """

        def __init__(self, dataloader: DataLoaderShard, device: torch.device):
            super().__init__(dataloader, device)
            self._rng_types = self._loader.rng_types
            self._loader.rng_types = None

        def __iter__(self):
            if self._rng_types is not None:
                synchronize_rng_states(self._rng_types, self._loader.synchronized_generator)

            return super().__iter__()

        @property
        def total_batch_size(self):
            return self._loader.total_batch_size

        @property
        def total_dataset_length(self):
            return self._loader.total_dataset_length

        @property
        def batch_sampler(self):
            return self._loader.batch_sampler



class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    """

    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset, skip_batches=0, **kwargs):
        super().__init__(dataset, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.skip_batches:
                yield batch


def skip_first_batches(dataloader, num_batches=0):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.
    """
    if is_torch_xla_available() and isinstance(dataloader, MpDeviceLoaderWrapper):
        dataloader._loader.device = dataloader._device
        dataloader = dataloader._loader

    dataset = dataloader.dataset
    sampler_is_batch_sampler = False
    if isinstance(dataset, IterableDataset):
        new_batch_sampler = None
    else:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = dataloader.batch_size

    if isinstance(dataloader, DataLoaderDispatcher):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        dataloader = DataLoaderDispatcher(
            dataset,
            split_batches=dataloader.split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader._drop_last,
            **kwargs,
        )
    elif isinstance(dataloader, DataLoaderShard):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        elif sampler_is_batch_sampler:
            kwargs["sampler"] = new_batch_sampler
            kwargs["batch_size"] = dataloader.batch_size
        else:
            kwargs["batch_sampler"] = new_batch_sampler
        dataloader = DataLoaderShard(
            dataset,
            device=dataloader.device,
            rng_types=dataloader.rng_types,
            synchronized_generator=dataloader.synchronized_generator,
            **kwargs,
        )
    else:
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            dataloader = SkipDataLoader(dataset, skip_batches=num_batches, **kwargs)
        else:
            dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

    if AcceleratorState().distributed_type == DistributedType.XLA:
        return MpDeviceLoaderWrapper2(dataloader, dataloader.device)
        
    return dataloader