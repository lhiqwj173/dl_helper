from dl_helper.train_param import match_num_processes, is_colab, is_kaggle
from dl_helper.tracker import Tracker, Tracker_None
from dl_helper.tracker import MODEL_FINAL, MODEL_BEST, MODEL_DUMMY, TEST_FINAL, TEST_BEST, TEST_DUMMY
from dl_helper.tool import report_memory_usage, check_nan, _check_nan
from dl_helper.acc.data_loader import skip_first_batches
from dl_helper.idx_manager import get_idx
from dl_helper.models.dummy import m_dummy

import copy
import traceback
import pickle
import shutil
import multiprocessing as mp

from tqdm import tqdm
import time, os, sys
from datetime import datetime
from datetime import timedelta
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

from py_ext.tool import log, debug, get_log_folder, _get_caller_info
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from py_ext.alist import alist

ses = os.environ.get('TG_SESSION')

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

from accelerate import Accelerator, load_checkpoint_in_model
from accelerate.utils import broadcast, InitProcessGroupKwargs
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
        # 拷贝 log 文件夹
        destination_folder = os.path.join(params.root, 'logs')
        source_folder = get_log_folder()
        os.makedirs(destination_folder, exist_ok=True)
        for file in os.listdir(source_folder):
            src = os.path.join(source_folder, file)
            target = os.path.join(destination_folder, file)
            # 覆盖拷贝文件
            shutil.copy(src, target)
        print('copy log folder done')

        zip_file = f'{params.root}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(params.root, zip_file, 9, inplace=False)
        print('compress_folder done')

        if not params.debug:
            # 上传更新到alist
            client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
            # 上传文件夹
            upload_folder = f'/{params.alist_upload_folder}/'
            client.mkdir(upload_folder)
            client.upload(zip_file, upload_folder)
        print('upload done')

    accelerator.wait_for_everyone()

last_checkpoint_time = 0
def checkpoint(epoch, accelerator, params, printer, need_check=True):
    global last_checkpoint_time
    if need_check:
        # 判断是否需要checkpoint
        need_checkpoint = torch.tensor(0, device=accelerator.device)
        if accelerator.is_main_process:
            # 20 min
            t = time.time()
            if t - last_checkpoint_time >= 60*20:
                need_checkpoint += 1
        accelerator.wait_for_everyone()
        need_checkpoint = broadcast(need_checkpoint)
    else:
        need_checkpoint = torch.tensor(1, device=accelerator.device)

    # 开始checkpoint
    if need_checkpoint.item() == 1:
        last_checkpoint_time = time.time()
        accelerator.save_state(os.path.join(params.root, 'checkpoint'))
        package_root(accelerator, params)

def train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, printer, trans, need_checkpoint=True):
    # 检查是否存在 step 记录
    skip_steps = tracker.step_count

    active_dataloader = train_loader
    # if skip_steps > 0:
    #     printer.print(f"[{epoch}] skipping train {skip_steps} steps.")
    #     active_dataloader = skip_first_batches(train_loader, skip_steps)

    model.train()
    for batch in active_dataloader:
        # 预处理
        data, target = trans(batch, train=True)

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
            # debug('track')
            tracker.track('train', output, target, loss)
            # debug('track done')

    # 追踪器，计算必要的数据
    tracker.update()
    # debug('update')

    # 缓存checkpoint
    if need_checkpoint:
        checkpoint(epoch, accelerator, params, printer, False)

    # # for debug
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     report_memory_usage(f"[{epoch}][{len(train_loader)}] train done")

def record_grad(idx, model, rank):
    with open(f'grad_{rank}.txt', 'a') as f:
        for param in model.parameters():
            f.write(f"step{idx}\ngrad: {param.grad}\nv: {param}\n\n")
            break

def print_grad(idx, model, printer):
    for param in model.parameters():
        printer.print(f"step{idx} grad: {param.grad} v: {param}", main=False)
        break

def test_train_func(data_file_path, id, test_class):
    test = test_class(idx=0)

    params = test.get_param()
    model = test.get_model()
    trans = test.get_transform(None)

    from .data import Dataset_cahce
    dataset = Dataset_cahce(params, 'test')# 使用test 避免类别均衡 导致拿不到数据
    dataset.files = [data_file_path]
    data_map = dataset._parse_data_map(dataset.files, 1, 0)
    dataset._load_data_map(data_map)

    idx = dataset.ids.index(id)
    batch = dataset.__getitem__(idx)
    batch = [i.unsqueeze(0).float() for i in batch]

    data, target = trans(batch, train=True)
    if not params.classify and len(target.shape) == 1:
        target = target.unsqueeze(1)
    output = model(data)
    batch_indices = _check_nan(output)

    print(f"[{idx}] batch_indices: {batch_indices}")


def train_fn_mini_epoch(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, printer, trans, need_checkpoint=True):
    # 检查是否存在 step 记录
    skip_steps = tracker.step_count

    active_dataloader = train_loader
    model.train()
    for mini_epoch in range(active_dataloader.sampler.mini_epoch):
        # 训练
        for batch in active_dataloader:
            # # 测试用
            # _batch = copy.deepcopy(batch)
            # debug(f'batch')
            # pickle.dump(batch, open(os.path.join(params.root, f'raw_batch_{accelerator.process_index}.pkl'), 'wb')) 
            # pickle.dump(active_dataloader.dataset.use_data_id, open(os.path.join(params.root, f'raw_ids_{accelerator.process_index}.pkl'), 'wb')) 

            # 预处理
            data, target = trans(batch, train=True)
            # debug(f'data :{data.shape} target :{target.shape}')
            # printer.print(f'data[0]: {data[0][:5][:5]}', main=False)

            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(target.shape) == 1:
                target = target.unsqueeze(1)
            # debug(f'unsqueeze')
                
            # record_grad(0, model, accelerator.process_index)
            optimizer.zero_grad()
            # debug(f'zero_grad')
            output = model(data)
            # debug(f'model')
            loss = criterion(output, target)
            # printer.print(f'loss: {loss}', main=False)
            # record_grad(1, model, accelerator.process_index)
            with torch.no_grad():
                check_nan(output, ids=active_dataloader.dataset.use_data_id)

            # debug(f'criterion')
            accelerator.backward(loss)
            # record_grad(2, model, accelerator.process_index)
            # debug(f'backward')
            optimizer.step()
            # record_grad(3, model, accelerator.process_index)
            # debug(f'step')

            # 追踪器 记录数据
            with torch.no_grad():
                # debug('track')
                tracker.track('train', output, target, active_dataloader, loss)
                # debug('track done')

        log(f"[{epoch}][{mini_epoch}] train done")

    # 追踪器，计算必要的数据
    tracker.update()
    # debug('update')

    # 缓存checkpoint
    if need_checkpoint:
        checkpoint(epoch, accelerator, params, printer, False)

def val_fn(epoch, params, model, criterion, val_data, accelerator, tracker, printer, trans):
    """
    异常模型在验证时checkpoint会报错, 默认不进行checkpoint
    """

    # 检查是否存在 step 记录
    skip_steps = tracker.step_count

    active_dataloader = val_data
    # if skip_steps > 0:
    #     printer.print(f"[{epoch}] skipping val {skip_steps} steps.")
    #     active_dataloader = skip_first_batches(val_data, skip_steps)

    model.eval()
    with torch.no_grad():
        for batch in active_dataloader:
            data, target = trans(batch)
            
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(target.shape) == 1:
                target = target.unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            # 追踪器 记录数据
            tracker.track('val', output, target, active_dataloader, loss)
    
    # debug('val loop done')

    # 追踪器，计算必要的数据
    tracker.update()
    # debug('val_fn done')

    # # for debug
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     report_memory_usage(f"[{epoch}][{len(val_data)}] val done")

def test_fn(params, model, blank_model, criterion, test_data, accelerator, tracker, printer, trans):
    test_types = [TEST_FINAL, TEST_BEST, TEST_DUMMY]
    models = [model]

    # 读取最佳模型
    # model_best = accelerator.unwrap_model(model)
    # load_checkpoint_in_model(model_best, os.path.join(params.root, MODEL_BEST))
    # models.append(model_best)
    load_checkpoint_in_model(blank_model, os.path.join(params.root, MODEL_BEST))
    models.append(blank_model)

    # dummy 模型
    printer.print(f'params.y_n: {params.y_n}')
    model_dummy = m_dummy(params.y_n)
    models.append(model_dummy)

    # 准备模型
    for i in range(2):
        models[i+1] = accelerator.prepare(models[i+1])

    for i, model in enumerate(models):
        printer.print(f'测试模型: {i}')

        model.eval()
        with torch.no_grad():
            for batch in test_data:
                data, target = trans(batch)

                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(target.shape) == 1:
                    target = target.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

                # 追踪器 记录数据
                tracker.track(test_types[i], output, target, test_data, loss)

        # 追踪器，计算必要的数据
        # printer.print('update')
        tracker.update()

    # for debug
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_memory_usage(f"test done")       

def output_fn(params, model, blank_model, criterion, train_loader, val_loader, accelerator, tracker, printer, trans):
    model_types = ['final', 'best']
    models = [model]

    # 读取最佳模型
    load_checkpoint_in_model(blank_model, os.path.join(params.root, MODEL_BEST))
    models.append(blank_model)
    # 准备模型
    models[1] = accelerator.prepare(models[1])

    data_loaders = [train_loader, val_loader]
    loader_names = ['train', 'val']

    for model_type, model in zip(model_types, models):
        printer.print(f'模型output: {model_type}')
        model.eval()
        with torch.no_grad():
            for i in range(len(data_loaders)):

                data_loader = data_loaders[i]
                loader_name = loader_names[i]
                printer.print(f'模型output: {model_type} {loader_name} 开始')

                run_type = f'{loader_name}_{model_type}'
                for mini_epoch in range(data_loader.sampler.mini_epoch):
                    for batch in data_loader:
                        data, target = trans(batch)

                        # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                        if not params.classify and len(target.shape) == 1:
                            target = target.unsqueeze(1)

                        output = model(data)

                        # 追踪器 记录数据
                        tracker.track(run_type, output, target, data_loader, None)

                # 追踪器，计算必要的数据
                # printer.print('update')
                tracker.update()

                # 等待同步
                accelerator.wait_for_everyone()
                printer.print(f'模型output: {model_type} {loader_name} 完成')

    # for debug
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_memory_usage(f"output done")       

def save_model_fn(params, model, accelerator, input_shape):
    accelerator.wait_for_everyone()
    accelerator.save_model(model, os.path.join(params.root, MODEL_FINAL))
    model = accelerator.unwrap_model(model)
    if accelerator.is_local_main_process:
        onnex_model_save_path = os.path.join(params.root, MODEL_FINAL, f'model.onnx')
        # 导出onnx
        try:
            torch.onnx.export(model, torch.randn(input_shape).to(accelerator.device), onnex_model_save_path, do_constant_folding=False,
            input_names=['input'], output_names=['output'])
        except Exception as e:
            log('导出onnx失败')
            log(e)

from dl_helper.models.binctabl import m_bin_ctabl

class printer():
    def __init__(self, lock, accelerator):
        self.lock = lock
        self.accelerator = accelerator
    
    def print(self, *msg, main=True):
        caller_info = _get_caller_info()
        head = f'[{self.accelerator.process_index}]'
        with self.lock:
            if main:
                if self.accelerator.is_local_main_process:
                    log(head, *msg, caller_info=caller_info)
            else:
                log(head, *msg, caller_info=caller_info)

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

def run_fn_cache_data(lock, num_processes, test_class, args, kwargs, train_param={}, model=None, only_predict=False):
    # 训练实例
    test = test_class(*args, **kwargs)
    try:

        # 训练参数
        params = test.get_param()
        set_seed(params.seed)

        accelerator = Accelerator(mixed_precision=params.amp if params.amp!='no' else 'no', kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))])
        p = printer(lock, accelerator)
        
        # 在root/title中添加 idx
        params.train_title = f'{params.train_title}_IDX{test.idx}'
        params.root = f'{params.root}_IDX{test.idx}'

        # 检查下载训练文件
        if (not params.debug) and accelerator.is_local_main_process:
            p.print('check alist download')
            
            client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
            try:
                _file = f'alist/{params.train_title}.7z'
                # 下载文件
                download_folder = f'/{params.alist_upload_folder}/'
                client.download(f'{download_folder}{params.train_title}.7z', 'alist/')
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
            else:
                os.makedirs(params.root, exist_ok=True)

        if params.debug:
            # 删除重建文件夹
            if os.path.exists(params.root):
                shutil.rmtree(params.root)
            os.makedirs(params.root, exist_ok=True)

        # 调整参数
        if num_processes >= 2:
            # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
            b = params.batch_size
            params.batch_size //= num_processes
            p.print(f'batch_size: {b} -> {params.batch_size}')
        
            if not params.abs_learning_rate:
                # 若不存在绝对学习率，需要基于设备调整lr
                l = params.learning_rate
                params.learning_rate *= num_processes
                p.print(f'learning_rate: {l} -> {params.learning_rate}')

        # 临时额外的训练参数
        if train_param:
            for k, v in train_param.items():
                setattr(params, k, v)
                p.print(f'{k}-> {v}')

        if None is model:
            model = test.get_model()

        if not only_predict:
            train_loader = test.get_cache_data('train', params, accelerator)
            val_loader = test.get_cache_data('val', params, accelerator)
            p.print(f'data init')

        # 绝对学习率优先
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate, weight_decay=params.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate,weight_decay=params.weight_decay)
        scheduler = test.get_lr_scheduler(optimizer, params)
        criterion = test.get_criterion()

        # # TEST
        # tracker = Tracker_None()
        # 训练跟踪
        tracker = Tracker(model.model_name(), params, accelerator, scheduler, num_processes, p)
        # 新增到 状态 管理
        accelerator.register_for_checkpointing(tracker)
        accelerator.register_for_checkpointing(scheduler)

        # 不需要准备数据
        if not only_predict:
            model, optimizer, scheduler = accelerator.prepare(
                model, optimizer, scheduler
            )
        else:
            model = accelerator.prepare(model)
        # model = model.to(accelerator.device)

        p.print(f'prepare done')

        # 数据增强
        trans = test.get_transform(accelerator.device)

        # 读取可能存在的训练数据（继续训练）
        checkpoint_folder = os.path.join(params.root, 'checkpoint')
        resume_from_checkpoint = os.path.exists(checkpoint_folder)
        if resume_from_checkpoint:
            accelerator.print(f"Resumed from checkpoint: {checkpoint_folder}")
            accelerator.load_state(checkpoint_folder)

            # 输出
            tracker.print_state()

        os.makedirs(os.path.join(params.root, MODEL_BEST), exist_ok=True)
        os.makedirs(os.path.join(params.root, MODEL_FINAL), exist_ok=True)
        os.makedirs(os.path.join(params.root, MODEL_DUMMY), exist_ok=True)
        
        # # for debug
        # tracker.need_test = True
        # only_predict = True

        # 训练循环
        if not only_predict:
            p.print(f'train start')
            for epoch in range(tracker.epoch_count, params.epochs):
                p.print(f'epoch {epoch} tracker.step_in_epoch: {tracker.step_in_epoch}')
                if tracker.step_in_epoch == 0:
                    # debug(f'train_fn_mini_epoch')
                    train_fn_mini_epoch(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, p, trans)

                # 验证
                p.print(f'epoch {epoch} val_fn')
                val_fn(epoch, params, model, criterion, val_loader, accelerator, tracker, p, trans)

                # 保存结果
                p.print(f'epoch {epoch} save_result')
                tracker.save_result()

                # 计算平均评价指标
                _max_mean_score_list = tracker.get_mean_socre_important()
                p.print(f'_max_mean_score_list:\n{_max_mean_score_list}')
                need_save_best_model, no_better_need_stop = torch.tensor(0, device=accelerator.device), torch.tensor(0, device=accelerator.device)
                if len(_max_mean_score_list) > 0:
                    _max_mean_f1 = max(_max_mean_score_list)
                    max_idx = _max_mean_score_list.index(_max_mean_f1)
                    if max_idx == len(_max_mean_score_list) - 1:
                        # 当前的模型版本最优
                        need_save_best_model += 1

                    if params.no_better_stop > 0 and (len(_max_mean_score_list) - 1 - max_idx) >= params.no_better_stop:
                        # 长时间无优化，停止训练
                        no_better_need_stop += 1

                # 同步
                accelerator.wait_for_everyone()
                need_save_best_model = broadcast(need_save_best_model)
                no_better_need_stop = broadcast(no_better_need_stop)
                p.print(f'need_save_best_model: {need_save_best_model}')
                p.print(f'no_better_need_stop: {no_better_need_stop}')
                if need_save_best_model:
                    # 记录最佳模型的 epoch
                    tracker.record_best_model_epoch()

                if (epoch % 30 == 0 and epoch > 0) or (need_save_best_model):

                    # 保存模型
                    p.print(f'epoch {epoch} save_model_fn')
                    save_model_fn(params, model, accelerator, test.get_in_out_shape()[0])

                    if need_save_best_model and accelerator.is_local_main_process:
                        # 拷贝记录最佳模型
                        p.print(f'epoch {epoch} save_model_bset')
                        model_folder = os.path.join(params.root, MODEL_FINAL)
                        best_folder = os.path.join(params.root, MODEL_BEST)
                        if os.path.exists(best_folder):
                            shutil.rmtree(best_folder)
                        shutil.copytree(model_folder, best_folder)

                # 打包
                # debug(f'package_root')
                package_root(accelerator, params)

                p.print(f'epoch {epoch} done')

                # 训练可用时长不足 / 早停
                # 开始 test/predict
                if tracker.need_test or no_better_need_stop:
                    break

        # 停止继续读取数据
        if not only_predict:
            p.print(f'close train data_loading')
            train_loader.sampler.data_loader_close()
            p.print(f'close val data_loading')
            val_loader.sampler.data_loader_close()

        p.print(f'test start')

        # 准备测试数据
        test_loader = test.get_cache_data('test', params, accelerator,predict_output=True)
        # 测试
        test_fn(params, model, test.get_model(), criterion, test_loader, accelerator, tracker, p, trans)

        # 保存模型
        save_model_fn(params, model, accelerator, test.get_in_out_shape()[0])

        # 绘图
        tracker.save_result()

        # 输出状态到日志
        tracker.print_state()

        # 输出模型预测，用于模型融合
        if params.need_meta_output:
            if not only_predict:
                train_loader = test.get_cache_data('train', params, accelerator, predict_output=True)
                val_loader = test.get_cache_data('val', params, accelerator,predict_output=True)
            output_fn(params, model, test.get_model(), criterion, train_loader, val_loader, accelerator, tracker, p, trans)

        # 打包
        package_root(accelerator, params)
        accelerator.wait_for_everyone()
    except Exception as e:
        exception_str = traceback.format_exc()
        wx.send_message(f'[{params.train_title}] 训练异常:\n{exception_str}')

        print(f'[{params.train_title}] 训练异常:\n{exception_str}', flush=True)

        print('pkill -f jupyter', flush=True)
        os.system('pkill -f jupyter')

        # # 方法1：停止当前cell的运行
        # print('sys.exit()', flush=True)
        # import sys
        # sys.exit()

        # # 方法2：中断内核
        # print('os._exit(0)', flush=True)
        # os._exit(0)

        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        os._exit(0)

        # 方法3：直接退出notebook
        print('HTML("<script>window.close();</script>")', flush=True)
        from IPython.core.display import HTML
        HTML("<script>window.close();</script>")

        # 方法4：重启内核
        print('kill_kernel()', flush=True)
        from IPython.kernel import kill_kernel
        kill_kernel()

        # 方法5：
        import IPython
        IPython.Application.instance().kernel.do_shutdown(False)

        raise e


def run_fn_1(lock, num_processes, test_class, args, kwargs, train_param={}, model=None, only_predict=False):
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
        
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
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
        else:
            os.makedirs(params.root, exist_ok=True)

    # 调整参数
    if num_processes >= 2:
        # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
        b = params.batch_size
        params.batch_size //= num_processes
        p.print(f'batch_size: {b} -> {params.batch_size}')
    
        if not params.abs_learning_rate:
            # 若不存在绝对学习率，需要基于设备调整lr
            l = params.learning_rate
            params.learning_rate *= num_processes
            p.print(f'learning_rate: {l} -> {params.learning_rate}')

    # 临时额外的训练参数
    if train_param:
        for k, v in train_param.items():
            setattr(params, k, v)
            p.print(f'{k}-> {v}')

    if None is model:
        model = test.get_model()

    if not only_predict:
        train_loader = test.get_data('train', params)
        val_loader = test.get_data('val', params)

        p.print(f'dataset length: {len(train_loader.dataset)}')
        p.print(f'dataloader length: {len(train_loader)}')

    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate,weight_decay=params.weight_decay)
    scheduler = test.get_lr_scheduler(optimizer, params)
    criterion = nn.CrossEntropyLoss()

    # # TEST
    # tracker = Tracker_None()
    # 训练跟踪
    tracker = Tracker(model.model_name(), params, accelerator, scheduler, num_processes, p)
    # 新增到 状态 管理
    accelerator.register_for_checkpointing(tracker)

    if not only_predict:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model = accelerator.prepare(
            model
        )

    p.print(f'prepare done')

    if not only_predict:
        # 数据增强
        trans = test.get_transform(accelerator.device)

    # 读取可能存在的训练数据（继续训练）
    checkpoint_folder = os.path.join(params.root, 'checkpoint')
    resume_from_checkpoint = os.path.exists(checkpoint_folder)
    if resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {checkpoint_folder}")
        accelerator.load_state(checkpoint_folder)
        # 输出
        tracker.print_state()

    need_xla_metrics_report = os.environ.get('XLA_METRICS_REPORT', "0") == '1'
    p.print(f'need_xla_metrics_report :{need_xla_metrics_report}')
    
    # 训练循环
    if not only_predict:
        for epoch in range(tracker.epoch_count, params.epochs):
            # 训练
            if tracker.step_in_epoch == 0:
                train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, p, trans)

            if need_xla_metrics_report and xm.is_master_ordinal():
                xm.rendezvous("train done")# mark_step
                with open(os.path.join(params.root, 'metrics_train.txt'), 'a') as f:
                    f.write(met.metrics_report())
                    f.write('\n---------------------------------------------\n')
                
            # 验证
            val_fn(epoch, params, model, criterion, val_loader, accelerator, tracker, p, trans)

            # 绘图
            tracker.save_result()

            # 打包
            package_root(accelerator, params)

            # 训练可用时长不足，开始 test/predict
            if tracker.need_test:
                break

        # 释放 训练/验证 数据集
        train_loader, val_loader = accelerator.clear(train_loader, val_loader)

    # 准备测试数据
    test_loader = test.get_data('test', params)
    test_loader = accelerator.prepare(test_loader)

    # 测试
    test_fn(params, model, criterion, test_loader, accelerator, tracker, p, trans)

    # 绘图
    tracker.save_result()

    # 打包
    package_root(accelerator, params)
    accelerator.wait_for_everyone()

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
    
        if not params.abs_learning_rate:
            # 若不存在绝对学习率，需要基于设备调整lr
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
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate,weight_decay=params.weight_decay)
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
    xm.master_print(f'ddp: {ddp}')
    xm.master_print(f'if_tqdm: {if_tqdm}')

    dist.init_process_group('xla', init_method='xla://')
    device = xm.xla_device()

    # 训练实例
    test = test_class(*args, **kwargs)

    # 训练参数
    params = test.get_param()

    # 调整参数
    if not params.abs_learning_rate:
        # 若不存在绝对学习率，需要基于设备调整lr
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
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate, weight_decay=params.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate,weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)

    trans = test.get_transform(device)

    xm.master_print(f'prepare done')
    xm.master_print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    xm.master_print(f'epochs: {params.epochs}')
    for epoch in range(params.epochs):
        model.train()

        activate_loader = train_loader if not if_tqdm else tqdm(train_loader, total=len(train_loader), disable=not xm.is_master_ordinal())
        for batch in activate_loader:# mark_step
            data, target = trans(batch, train=True)

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
        # xm.rendezvous("train done")# mark_step
        # if xm.is_master_ordinal():
        #     report_memory_usage(f"[{epoch}][{len(train_loader)}] train done")

        model.eval()
        with torch.no_grad():
            activate_loader = val_loader if not if_tqdm else tqdm(val_loader, total=len(val_loader), disable=not xm.is_master_ordinal())
            for batch in activate_loader:# mark_step
                data, target = trans(batch)
                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(target.shape) == 1:
                    target = target.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

        if epoch % 10 == 0:
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

def predict(test_class, *args, mode='normal', train_param={}, model=None, **kwargs):
    assert mode in ['normal', 'cache_data'], f'mode error: {mode}, must be normal / cache_data'
    num_processes = match_num_processes()
    lock = mp.Manager().Lock()

    if mode == 'cache_data':
        notebook_launcher(run_fn_cache_data, args=(lock, num_processes, test_class, args, kwargs, train_param, model, True), num_processes=num_processes)
    elif mode == 'normal':
        notebook_launcher(run_fn_1, args=(lock, num_processes, test_class, args, kwargs, train_param, model, True), num_processes=num_processes)

def run(test_class, *args, mode='normal', train_param={}, model=None, **kwargs):
    """
    mode: xla /xla_tqdm/simple/cache_data/ normal 
    args / kwargs 为tester构造参数

    可增加字典参数(都可在命令行添加):
        idx: 训练索引
        amp: 混合精度训练
        findbest_lr: 搜索学习率模式
        test: 测试运行, 设置epoch=10, 数据集取前4个数据文件

    """
    # # 测试用
    # kwargs['idx'] = 0

    # 分配idx
    from dl_helper.train_param import get_gpu_info
    base_title= f'{test_class.title_base()}_{get_gpu_info()}'
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('idx='):
                kwargs['idx'] = int(arg.split('=')[1])
            if arg.startswith('amp='):
                kwargs['amp'] = arg.split('=')[1]
            if arg.startswith('findbest_lr='):
                kwargs['findbest_lr'] = arg.split('=')[1]
            if arg == 'test':
                kwargs['test'] = True

    if 'findbest_lr' in kwargs: base_title+='_findbest_lr'
    if 'amp' in kwargs: base_title+=f'_{kwargs["amp"]}'
    if 'test' in kwargs and kwargs['test']:
        kwargs['idx'] = 0
    if 'idx' not in kwargs:
        kwargs['idx'] = get_idx(base_title)

    log(f'begin:{base_title} idx: {kwargs["idx"]}')

    num_processes = match_num_processes()

    os.environ['ALIST_USER'] = 'admin'
    os.environ['ALIST_PWD'] = 'LHss6632673'
    try:
        os.environ.pop('TPU_PROCESS_ADDRESSES')
        os.environ.pop('CLOUD_TPU_TASK_ID')
    except:
        pass

    lock = mp.Manager().Lock()

    if mode=='xla' and num_processes == 8:
        xmp.spawn(run_fn_xla, args=(lock, num_processes, test_class, args, kwargs, train_param, model, False), start_method='fork')     
    elif mode=='xla_tqdm' and num_processes == 8:
        xmp.spawn(run_fn_xla, args=(lock, num_processes, test_class, args, kwargs, train_param, model, True), start_method='fork')  
    elif mode == 'simple':
        notebook_launcher(run_fn, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)
    elif mode == 'cache_data':
        notebook_launcher(run_fn_cache_data, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)
    elif mode == 'normal':
        notebook_launcher(run_fn_1, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)
    else:
        raise Exception(f'mode error: {mode}, must be xla / xla_tqdm / simple / cache_data / normal')