from dl_helper.train_param import match_num_processes, is_colab, is_kaggle
from dl_helper.tracker import Tracker
from dl_helper.scheduler import ReduceLR_slow_loss
from dl_helper.tool import report_memory_usage
from dl_helper.acc.data_loader import skip_first_batches

import copy
import shutil
import multiprocessing as mp

from tqdm import tqdm
import time, os, sys
from datetime import datetime
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from dl_helper.tg import tg_download_async, tg_download, tg_upload, tg_del_file
from dl_helper.alist import alist

ses = os.environ.get('TG_SESSION')
user = os.environ.get('ALIST_USER')
pwd = os.environ.get('ALIST_PWD')

if match_num_processes() ==8:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla import runtime as xr
    from torch_xla.amp import autocast, GradScaler
    import torch_xla.debug.metrics as met
    try:
      from torch_xla.amp import syncfree
    except ImportError:
      assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"

from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import (
    PrecisionType,
    PrepareForLaunch,
    are_libraries_initialized,
    check_cuda_p2p_ib_support,
    get_gpu_info,
    is_mps_available,
    patch_environment,
    set_seed
)
def notebook_launcher(
    function,
    args=(),
    num_processes=None,
    mixed_precision="no",
    use_port="29500",
    master_addr="127.0.0.1",
    node_rank=0,
    num_nodes=1,
    rdzv_backend="static",
    rdzv_endpoint="",
    rdzv_conf=None,
    rdzv_id="none",
    max_restarts=0,
    monitor_interval=0.1,
):
    """
    Launches a training function, using several processes or multiple nodes if it's possible in the current environment
    (TPU with multiple cores for instance).

    <Tip warning={true}>

    To use this function absolutely zero calls to a CUDA device must be made in the notebook session before calling. If
    any have been made, you will need to restart the notebook and make sure no cells use any CUDA capability.

    Setting `ACCELERATE_DEBUG_MODE="1"` in your environment will run a test before truly launching to ensure that none
    of those calls have been made.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute. If it accepts arguments, the first argument should be the index of the
            process run.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*):
            The number of processes to use for training. Will default to 8 in Colab/Kaggle if a TPU is available, to
            the number of GPUs available otherwise.
        mixed_precision (`str`, *optional*, defaults to `"no"`):
            If `fp16` or `bf16`, will use mixed precision training on multi-GPU.
        use_port (`str`, *optional*, defaults to `"29500"`):
            The port to use to communicate between processes when launching a multi-GPU training.
        master_addr (`str`, *optional*, defaults to `"127.0.0.1"`):
            The address to use for communication between processes.
        node_rank (`int`, *optional*, defaults to 0):
            The rank of the current node.
        num_nodes (`int`, *optional*, defaults to 1):
            The number of nodes to use for training.
        rdzv_backend (`str`, *optional*, defaults to `"static"`):
            The rendezvous method to use, such as 'static' (the default) or 'c10d'
        rdzv_endpoint (`str`, *optional*, defaults to `""`):
            The endpoint of the rdzv sync. storage.
        rdzv_conf (`Dict`, *optional*, defaults to `None`):
            Additional rendezvous configuration.
        rdzv_id (`str`, *optional*, defaults to `"none"`):
            The unique run id of the job.
        max_restarts (`int`, *optional*, defaults to 0):
            The maximum amount of restarts that elastic agent will conduct on workers before failure.
        monitor_interval (`float`, *optional*, defaults to 0.1):
            The interval in seconds that is used by the elastic_agent as a period of monitoring workers.

    Example:

    ```python
    # Assume this is defined in a Jupyter Notebook on an instance with two GPUs
    from accelerate import notebook_launcher


    def train(*args):
        # Your training function here
        ...


    notebook_launcher(train, args=(arg1, arg2), num_processes=2, mixed_precision="fp16")
    ```
    """
    # Are we in a google colab or a Kaggle Kernel?
    in_colab = False
    in_kaggle = False
    if any(key.startswith("KAGGLE") for key in os.environ.keys()):
        in_kaggle = True
    elif "IPython" in sys.modules:
        in_colab = "google.colab" in str(sys.modules["IPython"].get_ipython())

    try:
        mixed_precision = PrecisionType(mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if (in_colab or in_kaggle) and (os.environ.get("TPU_WORKER_ID", None) is not None):
        # TPU launch
        import torch_xla.distributed.xla_multiprocessing as xmp

        if len(AcceleratorState._shared_state) > 0:
            raise ValueError(
                "To train on TPU in Colab or Kaggle Kernel, the `Accelerator` should only be initialized inside "
                "your training function. Restart your notebook and make sure no cells initializes an "
                "`Accelerator`."
            )
        if num_processes is None:
            num_processes = 8

        launcher = PrepareForLaunch(function, distributed_type="XLA")
        print(f"Launching a training on {num_processes} TPU cores.")
        xmp.spawn(launcher, args=args, nprocs=num_processes, start_method="fork")
    elif in_colab and get_gpu_info()[1] < 2:
        # No need for a distributed launch otherwise as it's either CPU or one GPU.
        if torch.cuda.is_available():
            print("Launching training on one GPU.")
        else:
            print("Launching training on one CPU.")
        function(*args)
    else:
        if num_processes is None:
            raise ValueError(
                "You have to specify the number of GPUs you would like to use, add `num_processes=...` to your call."
            )
        if node_rank >= num_nodes:
            raise ValueError("The node_rank must be less than the number of nodes.")
        if num_processes > 1:
            # Multi-GPU launch
            from torch.distributed.launcher.api import LaunchConfig, elastic_launch
            from torch.multiprocessing import start_processes
            from torch.multiprocessing.spawn import ProcessRaisedException

            if len(AcceleratorState._shared_state) > 0:
                raise ValueError(
                    "To launch a multi-GPU training from your notebook, the `Accelerator` should only be initialized "
                    "inside your training function. Restart your notebook and make sure no cells initializes an "
                    "`Accelerator`."
                )
            # Check for specific libraries known to initialize CUDA that users constantly use
            problematic_imports = are_libraries_initialized("bitsandbytes")
            if len(problematic_imports) > 0:
                err = (
                    "Could not start distributed process. Libraries known to initialize CUDA upon import have been "
                    "imported already. Please keep these imports inside your training function to try and help with this:"
                )
                for lib_name in problematic_imports:
                    err += f"\n\t* `{lib_name}`"
                raise RuntimeError(err)

            patched_env = dict(
                nproc=num_processes,
                node_rank=node_rank,
                world_size=num_nodes * num_processes,
                master_addr=master_addr,
                master_port=use_port,
                mixed_precision=mixed_precision,
            )

            # Check for CUDA P2P and IB issues
            if not check_cuda_p2p_ib_support():
                patched_env["nccl_p2p_disable"] = "1"
                patched_env["nccl_ib_disable"] = "1"

            # torch.distributed will expect a few environment variable to be here. We set the ones common to each
            # process here (the other ones will be set be the launcher).
            with patch_environment(**patched_env):
                # First dummy launch
                if os.environ.get("ACCELERATE_DEBUG_MODE", "false").lower() == "true":
                    launcher = PrepareForLaunch(test_launch, distributed_type="MULTI_GPU")
                    try:
                        start_processes(launcher, args=(), nprocs=num_processes, start_method="fork")
                    except ProcessRaisedException as e:
                        err = "An issue was found when verifying a stable environment for the notebook launcher."
                        if "Cannot re-initialize CUDA in forked subprocess" in e.args[0]:
                            raise RuntimeError(
                                f"{err}"
                                "This likely stems from an outside import causing issues once the `notebook_launcher()` is called. "
                                "Please review your imports and test them when running the `notebook_launcher()` to identify "
                                "which one is problematic and causing CUDA to be initialized."
                            ) from e
                        else:
                            raise RuntimeError(f"{err} The following error was raised: {e}") from e
                # Now the actual launch
                launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")
                print(f"Launching training on {num_processes} GPUs.")
                try:
                    if rdzv_conf is None:
                        rdzv_conf = {}
                    if rdzv_backend == "static":
                        rdzv_conf["rank"] = node_rank
                        if not rdzv_endpoint:
                            rdzv_endpoint = f"{master_addr}:{use_port}"
                    launch_config = LaunchConfig(
                        min_nodes=num_nodes,
                        max_nodes=num_nodes,
                        nproc_per_node=num_processes,
                        run_id=rdzv_id,
                        rdzv_endpoint=rdzv_endpoint,
                        rdzv_backend=rdzv_backend,
                        rdzv_configs=rdzv_conf,
                        max_restarts=max_restarts,
                        monitor_interval=monitor_interval,
                        start_method="fork",
                        # log_line_prefix_template=os.environ.get("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"),
                    )
                    elastic_launch(config=launch_config, entrypoint=function)(*args)
                except ProcessRaisedException as e:
                    if "Cannot re-initialize CUDA in forked subprocess" in e.args[0]:
                        raise RuntimeError(
                            "CUDA has been initialized before the `notebook_launcher` could create a forked subprocess. "
                            "This likely stems from an outside import causing issues once the `notebook_launcher()` is called. "
                            "Please review your imports and test them when running the `notebook_launcher()` to identify "
                            "which one is problematic and causing CUDA to be initialized."
                        ) from e
                    else:
                        raise RuntimeError(f"An issue was found when launching the training: {e}") from e

        else:
            # No need for a distributed launch otherwise as it's either CPU, GPU or MPS.
            if is_mps_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                print("Launching training on MPS.")
            elif torch.cuda.is_available():
                print("Launching training on one GPU.")
            else:
                print("Launching training on CPU.")
            function(*args)

def package_root(accelerator, params):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        zip_file = f'{params.root}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(params.root, zip_file, 9, inplace=False)

        if not params.debug:
            # # 删除当前的训练文件，如果存在
            # tg_del_file(ses, os.path.basename(zip_file))
            # # 上传到tg
            # tg_upload(ses, zip_file)

            # 上传更新到alist
            client = alist(user, pwd)
            client.upload(zip_file, '/train_data/')

    accelerator.wait_for_everyone()

def checkpoint(epoch, idx, accelerator, params, printer):
    printer.print(f"[{epoch}][{idx}] checkpointing...")
    accelerator.save_state(os.path.join(params.root, 'checkpoint'))
    # package_root(accelerator, params)
    accelerator.wait_for_everyone()
    printer.print(f"[{epoch}][{idx}] checkpointing done")

def train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, printer):
    # 检查是否存在 step 记录
    skip_steps = tracker.step_count

    active_dataloader = train_loader
    if skip_steps > 0:
        printer.print(f"[{epoch}] skipping train {skip_steps} steps.")
        active_dataloader = skip_first_batches(train_loader, skip_steps)

    model.train()
    for idx, (data, target) in tqdm(enumerate(active_dataloader), total=len(active_dataloader), disable=not accelerator.is_main_process, desc=f'[{epoch}] epoch train'):
        # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
        if not params.classify and len(target.shape) == 1:
            target = target.unsqueeze(1)
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        accelerator.backward(loss)
        optimizer.step()

        # 追踪器 记录数据
        with torch.no_grad():
            tracker.track(output, target, loss, 'train')

        # 缓存checkpoint
        if tracker.need_save:
            if idx % params.checkpointing_steps == 0:
                checkpoint(epoch, idx + skip_steps, accelerator, params, printer)

    # 追踪器，计算必要的数据
    tracker.update()

    # 缓存checkpoint
    checkpoint(epoch, idx + skip_steps, accelerator, params, printer)

    # for debug
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_memory_usage(f"[{epoch}][{len(train_loader)}] train done")

def val_fn(epoch, params, model, criterion, val_data, accelerator, tracker, printer, checkpoint=False):
    """
    异常模型在验证时checkpoint会报错, 默认不进行checkpoint
    """

    # 检查是否存在 step 记录
    skip_steps = tracker.step_count

    active_dataloader = val_data
    if skip_steps > 0:
        printer.print(f"[{epoch}] skipping val {skip_steps} steps.")
        active_dataloader = skip_first_batches(val_data, skip_steps)

    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(active_dataloader), total=len(active_dataloader), disable=not accelerator.is_main_process, desc=f'[{epoch}] epoch validating'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            # 追踪器 记录数据
            tracker.track(output, target, loss, 'val')

            # 缓存checkpoint
            if checkpoint and tracker.need_save:
                if idx % params.checkpointing_steps == 0:
                    checkpoint(epoch, idx + skip_steps, accelerator, params, printer)

    # 追踪器，计算必要的数据
    tracker.update()

    # for debug
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_memory_usage(f"[{epoch}][{len(val_data)}] val done")

def test_fn(params, model, criterion, test_data, accelerator, tracker, printer):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(test_data), total=len(test_data), disable=not accelerator.is_main_process, desc=f'test model'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(target.shape) == 1:
                target = target.unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            # 追踪器 记录数据
            tracker.track(output, target, loss, 'test')

    # 追踪器，计算必要的数据
    tracker.update()

    # for debug
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_memory_usage(f"test done")       


from dl_helper.models.binctabl import m_bin_ctabl

class printer():
    def __init__(self, lock, accelerator):
        self.lock = lock
        self.accelerator = accelerator
    
    def print(self, *msg, main=True, **kwargs):
        head = f'[{self.accelerator.process_index}]'
        with self.lock:
            if main:
                self.accelerator.print(head, *msg, **kwargs)
            else:
                print(head, *msg, **kwargs)

def get_data_sampler(data_set, _type):
    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True if _type == 'train' else False
            )

    return train_sampler

def run_fn_1(lock, num_processes, test_class, args, kwargs, train_param={}, model=None):
    set_seed(42)

    # 训练实例
    test = test_class(*args, **kwargs)

    # 训练参数
    params = test.get_param()

    accelerator = Accelerator(mixed_precision=params.amp if params.amp!='no' else 'no')
    p = printer(lock, accelerator)
    
    # 检查下载训练文件
    if (not params.debug) and accelerator.is_local_main_process:
        p.print('check alist download')
        
        client = alist(user, pwd)
        try:
            _file = f'alist/{params.train_title}.7z'
            # 下载文件
            client.download(f'/train_data/{params.train_title}.7z', 'alist/')
            p.print(f'download {_file}')

        except:
            pass

        if os.path.exists(_file):
            # 解压文件
            decompress(_file)
            p.print(f'decompress {_file}')
            # move 
            folder = os.path.join('/kaggle/working/alist', params.train_title, 'checkpoint')
            p.print(f'checkpoint folder {folder}')
            if os.path.exists(folder):
                wx.send_message(f'[{params.train_title}] 使用alist缓存文件继续训练')
                p.print(f"使用alist缓存文件继续训练")
                shutil.copytree(os.path.join('/kaggle/working/alist', params.train_title), params.root, dirs_exist_ok=True)

    ## 检查下载tg训练文件
    # if (not params.debug) and accelerator.is_local_main_process:
    #     p.print('check tg download')
    #     tg_download(
    #         ses,
    #         f'{params.train_title}.7z',
    #         '/kaggle/working/tg'
    #     )

    #     # 如果存在 checkpoints ，拷贝到正确的路径以继续训练
    #     folder = os.path.join('/kaggle/working/tg', params.train_title, 'checkpoint')
    #     p.print(f'folder: {folder}')
    #     if os.path.exists(folder):
    #         wx.send_message(f'[{params.train_title}] 使用tg缓存文件继续训练')
    #         p.print(f"使用tg缓存文件继续训练")
    #         shutil.copytree(os.path.join('/kaggle/working/tg', params.train_title), params.root, dirs_exist_ok=True)

    # 调整参数
    if num_processes >= 2:
        # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
        b = params.batch_size
        params.batch_size //= num_processes
        p.print(f'batch_size: {b} -> {params.batch_size}')
    
        # 调整lr
        l = params.learning_rate
        params.learning_rate *= num_processes
        p.print(f'learning_rate: {l} -> {params.learning_rate}')

    p.print(f'batch_size: {params.batch_size}')

    # 临时额外的训练参数
    if train_param:
        for k, v in train_param.items():
            setattr(params, k, v)

    train_loader = test.get_data('train', params)
    val_loader = test.get_data('val', params)

    p.print(f'dataset length: {len(train_loader.dataset)}')
    p.print(f'dataloader length: {len(train_loader)}')

    if None is model:
        model = test.get_model()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate,weight_decay=params.weight_decay)
    scheduler = ReduceLR_slow_loss(optimizer)

    # 训练跟踪
    tracker = Tracker(params, accelerator, scheduler, num_processes, p)
    # 新增到 状态 管理
    accelerator.register_for_checkpointing(tracker)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    p.print(f'prepare done')
    p.print(f'each epoch step: {len(train_loader)}')

    # 读取可能存在的训练数据（继续训练）
    checkpoint_folder = os.path.join(params.root, 'checkpoint')
    resume_from_checkpoint = os.path.exists(checkpoint_folder)
    if resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {checkpoint_folder}")
        accelerator.load_state(checkpoint_folder)
        # 输出
        tracker.print_state()
    
    # 训练循环
    for epoch in range(tracker.epoch_count, params.epochs):
        # 训练
        if tracker.step_in_epoch == 0:
            train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, p)
            
        # 验证
        val_fn(epoch, params, model, criterion, val_loader, accelerator, tracker, p)

        # 绘图
        tracker.plot()

        # 打包
        package_root(accelerator, params)

    # 释放 训练/验证 数据集 optimizer
    optimizer, train_loader, val_loader = accelerator.clear(optimizer, train_loader, val_loader)

    # 准备测试数据
    test_loader = test.get_data('test', params)
    test_loader = accelerator.prepare(test_loader)

    # 测试
    test_fn(params, model, criterion, test_loader, accelerator, tracker, p)

    # 绘图
    tracker.plot()

    # 打包
    package_root(accelerator, params)

def run_fn(lock, num_processes, test_class, args, kwargs, train_param={}, model=None):
    set_seed(42)

    # 训练实例
    test = test_class(*args, **kwargs)

    # 训练参数
    params = test.get_param()

    accelerator = Accelerator(mixed_precision=params.amp if params.amp!='no' else 'no')
    p = printer(lock, accelerator)

    # 调整参数
    if num_processes >= 2:
        # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
        b = params.batch_size
        params.batch_size //= num_processes
        p.print(f'batch_size: {b} -> {params.batch_size}')
    
    # if num_processes > 1:
        # 调整lr
        l = params.learning_rate
        params.learning_rate *= num_processes
        p.print(f'learning_rate: {l} -> {params.learning_rate}')

    p.print(f'batch_size: {params.batch_size}')

    if train_param:
        for k, v in train_param.items():
            setattr(params, k, v)

    train_loader = test.get_data('train', params)
    val_loader = test.get_data('val', params)

    p.print(f'dataset length: {len(train_loader.dataset)}')
    p.print(f'dataloader length: {len(train_loader)}')

    if accelerator.is_main_process:
        report_memory_usage(f'init train data done')

    if None is model:
        model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate,weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    p.print(f'prepare done')
    p.print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    for epoch in range(params.epochs):
        model.train()
        for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_main_process):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(target.shape) == 1:
                target = target.unsqueeze(1)
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            accelerator.backward(loss)
            optimizer.step()
        
        scheduler.step()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            report_memory_usage(f"[{epoch}][{len(train_loader)}] train done")

        model.eval()
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader), disable=not accelerator.is_main_process):
                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(target.shape) == 1:
                    target = target.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            report_memory_usage(f"[{epoch}][{len(val_loader)}] val done")

    if accelerator.is_main_process:
        report_memory_usage('all done')

def run_fn_xla(index, lock, num_processes, test_class, args, kwargs, train_param={}, model=None, if_tqdm=False):
    ddp = True if is_kaggle() else False
    dist.init_process_group('xla', init_method='xla://')
    device = xm.xla_device()

    # 训练实例
    test = test_class(*args, **kwargs)

    # 训练参数
    params = test.get_param()

    # 调整参数
    # 调整lr
    l = params.learning_rate
    params.learning_rate *= num_processes
    xm.master_print(f'learning_rate: {l} -> {params.learning_rate}')
    params.batch_size //= num_processes
    xm.master_print(f'batch_size: {params.batch_size}')

    if train_param:
        for k, v in train_param.items():
            setattr(params, k, v)

    # 真实数据
    train_loader = test.get_data('train', params, get_data_sampler)
    val_loader = test.get_data('val', params, get_data_sampler)

    xm.rendezvous("init train_loader")# mark_step
    xm.master_print(f'dataset length: {len(train_loader.dataset)}')
    xm.master_print(f'dataloader length: {len(train_loader)}')

    if xm.is_master_ordinal():
        report_memory_usage(f'init train data done')
        if not os.path.exists(params.root):
            shutil.rmtree(params.root, ignore_errors=True)
        os.makedirs(params.root, exist_ok=True)

    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)

    if None is model:
        model = test.get_model()
    model = model.to(device)
    if ddp:
        if xr.using_pjrt():
            xm.master_print('broadcast_master_param')
            xm.broadcast_master_param(model)
        model = DDP(model, gradient_as_bucket_view=True)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate,weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)

    xm.master_print(f'prepare done')
    xm.master_print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    for epoch in range(params.epochs):
        model.train()

        activate_loader = train_loader if not if_tqdm else tqdm(train_loader, total=len(train_loader), disable=not xm.is_master_ordinal())
        for data, target in activate_loader:# mark_step
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(target.shape) == 1:
                target = target.unsqueeze(1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if ddp:
                optimizer.step()
            else:
                xm.optimizer_step(optimizer)

        xm.rendezvous("train done")# mark_step
        if xm.is_master_ordinal():
            with open(os.path.join(params.root, 'metrics_train.txt'), 'a') as f:
                f.write(met.metrics_report())
                f.write('\n---------------------------------------------\n')

        scheduler.step()
        xm.rendezvous("train done")# mark_step
        if xm.is_master_ordinal():
            report_memory_usage(f"[{epoch}][{len(train_loader)}] train done")

        model.eval()
        with torch.no_grad():
            activate_loader = val_loader if not if_tqdm else tqdm(val_loader, total=len(val_loader), disable=not xm.is_master_ordinal())
            for data, target in activate_loader:# mark_step
                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(target.shape) == 1:
                    target = target.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

        xm.rendezvous("val done")# mark_step
        if xm.is_master_ordinal():
            report_memory_usage(f"[{epoch}][{len(val_loader)}] val done")

    if xm.is_master_ordinal():
        with open(os.path.join(params.root, 'metrics_report_epoch.txt'), 'a') as f:
            f.write(met.metrics_report())

        if os.path.exists('hlo'):
            # 移动到 params.root
            shutil.copytree('hlo', os.path.join(params.root, 'hlo'))

        compress_folder(params.root, params.root + '.7z', 9, inplace=False)

        report_memory_usage('all done')

def test_func():
    acc = Accelerator()
    
    # 线性模型
    model = nn.Linear(10, 2)

    # 模拟数据
    data_length = 1000
    data = torch.randn(data_length, 10)
    target = torch.randint(0, 2, (data_length,))
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, target), batch_size=2, shuffle=True)

    # validation
    data_length = 100
    data = torch.randn(data_length, 10)
    target = torch.randint(0, 2, (data_length,))
    val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, target), batch_size=2, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # 训练
    model, train_dataloader, val_dataloader, optimizer = acc.prepare(model, train_dataloader, val_dataloader, optimizer)

    acc.print(f'开始训练')
    for i in range(10):
        # 训练
        model.train()
        for idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            acc.backward(loss)
            optimizer.step()

            acc.print(f'{i} {idx} train checkpoint...')
            acc.save_state('checkpoint')
            acc.print(f'{i} {idx} train checkpoint done')

        # 验证
        model.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(val_dataloader):
                output = model(data)
                loss = criterion(output, target)

                acc.print(f'{i} {idx} val checkpoint...')
                acc.save_state('checkpoint')
                acc.print(f'{i} {idx} val checkpoint done')


def run(test_class, *args, mode='', train_param={}, model=None, **kwargs):
    """
    mode: xla / normal / simple
    """
    num_processes = match_num_processes()

    # model = None
    # if num_processes == 8:
    #     # tpu 在训练函数外实例化模型 传入
    #     print('训练函数外实例化模型 传入')
    #     model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)

    lock = mp.Manager().Lock()

    if num_processes == 8:
        try:
            os.environ.pop('CLOUD_TPU_TASK_ID')
            os.environ.pop('TPU_PROCESS_ADDRESSES')
            # os.environ['ACCELERATE_DEBUG_MODE'] = '1'
        except:
            pass

    if mode=='xla' and num_processes == 8:
        xmp.spawn(run_fn_xla, args=(lock, num_processes, test_class, args, kwargs, train_param, model, False), start_method='fork')     
    elif mode=='xla_tqdm' and num_processes == 8:
        xmp.spawn(run_fn_xla, args=(lock, num_processes, test_class, args, kwargs, train_param, model, True), start_method='fork')  
    elif mode == 'simple':
        notebook_launcher(run_fn, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)
    else:
        notebook_launcher(run_fn_1, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)