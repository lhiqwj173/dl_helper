from dl_helper.train_param import match_num_processes
from dl_helper.tracker import Tracker
from dl_helper.scheduler import ReduceLR_slow_loss
from dl_helper.tool import report_memory_usage
from dl_helper.trainers.gpu import train_gpu
from dl_helper.trainers.tpu import train_tpu

import copy

import multiprocessing as mp

from tqdm import tqdm
import time, os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

if match_num_processes() ==8:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla import runtime as xr
    from torch_xla.amp import autocast, GradScaler
    try:
      from torch_xla.amp import syncfree
    except ImportError:
      assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"

from accelerate import notebook_launcher

def train_fn(index, epoch, params, model, criterion, optimizer, train_data, trainer, tracker):
    model.train()
    for idx, (data, target) in tqdm(enumerate(train_data), total=len(train_data), disable=not trainer.is_main_process(), desc=f'epoch {epoch} training'):
        # trainer.wait_for_everyone()
        # if trainer.is_main_process():
        #     report_memory_usage(f'begin')
        
        # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
        if not params.classify and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        data, target = trainer.prepare(data, target)
        optimizer.zero_grad()
        output, loss = trainer.cal_output_loss(model, data, target, criterion)

        # tracker.track(output, target, loss, 'train')
        trainer.step(loss, optimizer)

        if idx % 10 == 0:
            if trainer.is_main_process():
                report_memory_usage(f'step')
        trainer.wait_for_everyone()

def val_fn(index, epoch, params, model, val_data, trainer, criterion, tracker):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_data), total=len(val_data), disable=not trainer.is_main_process(), desc=f'epoch {epoch} validating'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            data, target = trainer.prepare(data, target)
            output, loss = trainer.cal_output_loss(model, data, target, criterion)
            # tracker.track(output, target, loss, 'val')

            trainer.wait_for_everyone()
            if trainer.is_main_process():
                report_memory_usage(f'val_{idx}')

def test_fn(index, params, model, test_data, trainer):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_data):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            data, target = trainer.prepare(data, target)
            output = model(data)

def run_fn_0(index, lock, num_processes, test):
    ###########################################
    # 1. 训练/验证
    ###########################################
    # 调整参数
    params = test.get_param()
    # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    # 创建训练器
    if num_processes == 8:
        trainer = train_tpu(params, index, lock)
    else:
        trainer = train_gpu(params, index, lock)
    device = trainer.get_device()

    trainer.print('准备训练元素...')
    model = test.get_model()
    train_data = test.get_data('train', params, trainer.get_data_sampler)
    val_data = test.get_data('val', params, trainer.get_data_sampler)
    criterion = test.get_criterion()
    optimizer = test.get_optimizer(model)
    scheduler = ReduceLR_slow_loss(optimizer)
    trainer.print('done')
                
    # 训练追踪器
    tracker = Tracker(params, trainer, scheduler, num_processes)

    trainer.print('初始化训练元素...')
    model = trainer.init_model(model)
    trainer.print('model')
    train_data = trainer.init_data_loader(train_data)
    trainer.print('train_data')
    val_data = trainer.init_data_loader(val_data)
    trainer.print('val_data')
    criterion = trainer.init_criterion(criterion)
    trainer.print('criterion')
    scheduler = trainer.init_scheduler(scheduler)
    trainer.print('scheduler')
    tracker = trainer.init_tracker(tracker)
    trainer.print('done')

    # 同步
    if trainer.is_main_process():
        report_memory_usage('开始训练')

    # for epoch in tqdm(range(params.epochs), disable=not trainer.is_main_process()):
    for epoch in range(params.epochs):
        # 训练
        train_fn(index, epoch, params, model, criterion, optimizer, train_data, trainer, tracker)
        # 同步
        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage()

        # 验证
        val_fn(index, epoch, params, model, val_data, trainer, criterion, tracker)
        # 同步
        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage()

        # 每epoch更新
        tracker.update()

        # 缓存训练数据
        if tracker.need_save:
            trainer.print('cache train data')
            tracker.save()

        # 同步
        trainer.wait_for_everyone()

    return

    ###########################################
    # 1. 测试
    ###########################################
    trainer.print('开始测试')
    trainer.clear_data_loader()
    del train_data, val_data, optimizer

    test_data = test.get_data('test', params, trainer.get_data_sampler)
    test_data = trainer.init_data_loader(test_data)

    val_fn(index, params, model, test_data, trainer, criterion, tracker)
    
    tracker.update()
    tracker.plot()


def get_data_sampler(data_set):
    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

    return train_sampler

def run_fn_1(index, lock, num_processes, test):
    ###########################################
    # 1. 训练/验证
    ###########################################
    # 调整参数
    params = test.get_param()
    # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    dist.init_process_group('xla', init_method='xla://')

    device = xm.xla_device()

    xm.master_print('准备训练元素...')
    model = test.get_model()
    xm.master_print('model')
    model = model.to(device)
    if xr.using_pjrt():
        xm.broadcast_master_param(model)
    model = DDP(model, gradient_as_bucket_view=True)

    train_data = test.get_data('train', params, get_data_sampler)
    val_data = test.get_data('val', params, get_data_sampler)
    criterion = test.get_criterion()
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = ReduceLR_slow_loss(optimizer)
    xm.master_print('done')

    xm.master_print('train_data')
    train_data = pl.MpDeviceLoader(train_data, device)
    xm.master_print('val_data')
    val_data = pl.MpDeviceLoader(val_data, device)
    xm.master_print('done')

    if xm.is_master_ordinal():
        report_memory_usage('开始训练')
  
    for epoch in range(params.epochs):
        # 训练
        model.train()
        for idx, (data, target) in tqdm(enumerate(train_data), total=len(train_data), disable=not xm.is_master_ordinal(), desc=f'epoch {epoch} training'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if idx % 3 == 0:
                if xm.is_master_ordinal():
                    report_memory_usage(f'step {idx}')

            # xm.mark_step()

        # 同步
        xm.rendezvous("wait_for_everyone")

        # 验证
        model.eval()
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(val_data), total=len(val_data), disable=not xm.is_master_ordinal(), desc=f'epoch {epoch} validating'):
                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

                if idx % 3 == 0:
                    if xm.is_master_ordinal():
                        report_memory_usage(f'step {idx}')

                # xm.mark_step()

        # 同步
        xm.rendezvous("wait_for_everyone")

        # # 每epoch更新
        # tracker.update()

        # # 缓存训练数据
        # if tracker.need_save:
        #     xm.master_print('cache train data')
        #     # tracker.save()


    return

    ###########################################
    # 1. 测试
    ###########################################
    xm.master_print('开始测试')
    trainer.clear_data_loader()
    del train_data, val_data, optimizer

    test_data = test.get_data('test', params, trainer.get_data_sampler)
    test_data = trainer.init_data_loader(test_data)

    val_fn(index, params, model, test_data, trainer, criterion, tracker)
    
    tracker.update()
    tracker.plot()

from dl_helper.models.binctabl import m_bin_ctabl

def run_fn(index, lock, num_processes, test):
    # 设置训练参数
    epochs = 30

    dist.init_process_group('xla', init_method='xla://')

    device = xm.xla_device()

    batch_size = 1024

    # 创建模拟数据
    num_classes = 3

    # TPU limit
    num_samples = 3000000 
    
    # P100 limit
    num_samples = 2400000 
    
    # T4x2 limit
    num_samples = 1100000 
    
    # for debug
    num_samples = 50000

    data = torch.randn(num_samples, 40, 100)
    target = torch.randint(0, num_classes, (num_samples,))
    train_dataset = torch.utils.data.TensorDataset(data, target)

    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True
        # shuffle=True,
    )

    train_loader = pl.MpDeviceLoader(train_loader, device)

    # model = ResNet()
    model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()

    # 初始化优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # TPU  each epoch step: 782
    # P100 each epoch step: 6250
    # T4*2 each epoch step: 3125
    xm.master_print(f'batch_size: {batch_size}')
    xm.master_print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    for epoch in range(epochs):
        for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), disable=not xm.is_master_ordinal()):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        if xm.is_master_ordinal():
            report_memory_usage(f'{epoch}')

def run(test):
    # bug:
    # WARNING: Logging before InitGoogle() is written to STDERR
    # WARNING: Logging before InitGoogle() is written to STDERR
    # E0000 00:00:1719236516.757616   18445 common_lib.cc:778] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8477 in any of the 4 ports provided in `tpu_process_addresses`="localhost:8476,localhost:8477,localhost:8478,localhost:8479"
    # === Source Location Trace: === 
    # learning/45eac/tfrc/runtime/common_lib.cc:467
    # E0000 00:00:1719236516.757613   18459 common_lib.cc:778] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8479 in any of the 4 ports provided in `tpu_process_addresses`="localhost:8476,localhost:8477,localhost:8478,localhost:8479"
    # === Source Location Trace: ===
    # learning/45eac/tfrc/runtime/common_lib.cc:467
    # WARNING: Logging before InitGoogle() is written to STDERR
    # E0000 00:00:1719236516.758186   18452 common_lib.cc:778] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8478 in any of the 4 ports provided in `tpu_process_addresses`="localhost:8476,localhost:8477,localhost:8478,localhost:8479"
    # === Source Location Trace: ===
    # learning/45eac/tfrc/runtime/common_lib.cc:467
    # E0624 13:41:56.788773396   18445 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-06-24T13:41:56.788753605+00:00"}
    # E0624 13:41:56.789400842   18452 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-06-24T13:41:56.789384548+00:00", grpc_status:2}
    # E0624 13:41:56.789423994   18459 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-06-24T13:41:56.789409058+00:00", grpc_status:2}
    # E0624 13:41:56.795029596   18438 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-06-24T13:41:56.795014401+00:00", grpc_status:2}
    # [W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
    # [W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
    # [W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
    # 准备训练元素...
    # model
    # https://symbolize.stripped_domain/r/?trace=7e04d17dcc2d,7e070ddaf04f,7e04d17dccf4,7e04d17dd00c,7e04d142d4ca,7e04d144c65f,7e04d16c0352,7e06e798a817&map= 
    # *** SIGSEGV (@0x18), see go/stacktraces#s15 received by PID 18459 (TID 19933) on cpu 66; stack trace: ***
    # PC: @     0x7e04d17dcc2d  (unknown)  std::_Hashtable<>::_M_find_before_node()
    #     @     0x7e035f101067        928  (unknown)
    #     @     0x7e070ddaf050       3392  (unknown)
    #     @     0x7e04d17dccf5         80  std::__detail::_Map_base<>::operator[]()
    #     @     0x7e04d17dd00d        496  torch_xla::xla_cpu_fallback()
    #     @     0x7e04d142d4cb        112  at::native::_call_fallback_fn<>::call()
    #     @     0x7e04d144c660        112  torch_xla::XLANativeFunctions::_local_scalar_dense()
    #     @     0x7e04d16c0353         96  c10::impl::make_boxed_from_unboxed_functor<>::call()
    #     @     0x7e06e798a818         48  c10::Dispatcher::callBoxed()
    #     @ ... and at least 1 more frames
    # https://symbolize.stripped_domain/r/?trace=7e04d17dcc2d,7e035f101066,7e070ddaf04f,7e04d17dccf4,7e04d17dd00c,7e04d142d4ca,7e04d144c65f,7e04d16c0352,7e06e798a817&map= 
    # E0624 13:41:59.618505   19933 coredump_hook.cc:442] RAW: Remote crash data gathering hook invoked.
    # E0624 13:41:59.618516   19933 client.cc:269] RAW: Coroner client retries enabled (b/136286901), will retry for up to 30 sec.
    # E0624 13:41:59.618521   19933 coredump_hook.cc:537] RAW: Sending fingerprint to remote end.
    # E0624 13:41:59.618552   19933 coredump_hook.cc:546] RAW: Cannot send fingerprint to Coroner: [NOT_FOUND] stat failed on crash reporting socket /var/google/services/logmanagerd/remote_coredump.socket (Is the listener running?): No such file or directory
    # E0624 13:41:59.618557   19933 coredump_hook.cc:598] RAW: Dumping core locally.
    # E0624 13:42:11.279552   19933 process_state.cc:807] RAW: Raising signal 11 with default behavior
    # https://symbolize.stripped_domain/r/?trace=7e070ddf8e96,7e070ddaf04f&map= 
    # https://symbolize.stripped_domain/r/?trace=7e070ddf8e96,*** SIGTERM received by PID 18438 (TID 18438) on cpu 93 from PID 13; stack trace: ***
    # 7e070ddaf04f&map= 
    # *** SIGTERM received by PID 18452 (TID 18452) on cpu 95 from PID 13; stack trace: ***
    # https://symbolize.stripped_domain/r/?trace=7e070ddf8e96,7e070ddaf04f&map= 
    # *** SIGTERM received by PID 18445 (TID 18445) on cpu 22 from PID 13; stack trace: ***
    # PC: @     0x7e070ddf8e96  (unknown)  (unknown)
    #     @     0x7e0357101067        928  (unknown)
    #     @     0x7e070ddaf050  (unknown)  (unknown)
    # https://symbolize.stripped_domain/r/?trace=7e070ddf8e96,7e0357101066,7e070ddaf04f&map= 
    # E0624 13:42:11.955676   18445 coredump_hook.cc:388] RAW: Remote crash gathering disabled for SIGTERM.
    # PC: @     0x7e070ddf8e96  (unknown)  (unknown)
    # PC: @     0x7e070ddf8e96  (unknown)  (unknown)
    #     @     0x7e035f101067        928  (unknown)
    #     @     0x7e035f101067        928  (unknown)
    #     @     0x7e070ddaf050  (unknown)  (unknown)
    #     @     0x7e070ddaf050  (unknown)  (unknown)
    # https://symbolize.stripped_domain/r/?trace=https://symbolize.stripped_domain/r/?trace=7e070ddf8e96,7e070ddf8e96,7e035f101066,7e035f101066,7e070ddaf04f7e070ddaf04f&map=&map= 
    
    # E0624 13:42:11.956408   18438 coredump_hook.cc:388] RAW: Remote crash gathering disabled for SIGTERM.
    # E0624 13:42:11.956408   18452 coredump_hook.cc:388] RAW: Remote crash gathering disabled for SIGTERM.
    # E0624 13:42:12.061995   18445 process_state.cc:807] RAW: Raising signal 15 with default behavior
    # E0624 13:42:12.064539   18452 process_state.cc:807] RAW: Raising signal 15 with default behavior
    # E0624 13:42:12.070060   18438 process_state.cc:807] RAW: Raising signal 15 with default behavior
    # ---------------------------------------------------------------------------
    # BrokenProcessPool                         Traceback (most recent call last)
    # File <timed eval>:1

    # File /kaggle/working/3rd/dl_helper/dl_helper/trainer.py:318, in run(test)
    #     315 lock = mp.Manager().Lock()
    #     317 if num_processes == 8:
    # --> 318     xmp.spawn(run_fn, args=(lock, num_processes, test), start_method='fork')      
    #     319 else:
    #     320     notebook_launcher(run_fn, args=(0, lock, num_processes, copy.deepcopy(test)), num_processes=num_processes)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/runtime.py:87, in requires_pjrt.<locals>.wrapper(*args, **kwargs)
    #     83 if not using_pjrt():
    #     84   raise NotImplementedError('`{}` not implemented for XRT'.format(
    #     85       fn.__name__))
    # ---> 87 return fn(*args, **kwargs)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/distributed/xla_multiprocessing.py:38, in spawn(fn, args, nprocs, join, daemon, start_method)
    #     6 @xr.requires_pjrt
    #     7 def spawn(fn,
    #     8           args=(),
    # (...)
    #     11           daemon=False,
    #     12           start_method='spawn'):
    #     13   """Enables multi processing based replication.
    #     14 
    #     15   Args:
    # (...)
    #     36     return None.
    #     37   """
    # ---> 38   return pjrt.spawn(fn, nprocs, start_method, args)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py:200, in spawn(fn, nprocs, start_method, args)
    #     197 elif nprocs is not None:
    #     198   logging.warning('Unsupported nprocs (%d), ignoring...' % nprocs)
    # --> 200 run_multiprocess(spawn_fn, start_method=start_method)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/runtime.py:87, in requires_pjrt.<locals>.wrapper(*args, **kwargs)
    #     83 if not using_pjrt():
    #     84   raise NotImplementedError('`{}` not implemented for XRT'.format(
    #     85       fn.__name__))
    # ---> 87 return fn(*args, **kwargs)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py:160, in run_multiprocess(fn, start_method, *args, **kwargs)
    #     154   mp_fn = functools.partial(
    #     155       _run_thread_per_device,
    #     156       local_world_size=num_processes,
    #     157       fn=functools.partial(fn, *args, **kwargs),
    #     158       initializer_fn=initialize_multiprocess)
    #     159   process_results = executor.map(mp_fn, range(num_processes))
    # --> 160   replica_results = list(
    #     161       itertools.chain.from_iterable(
    #     162           result.items() for result in process_results))
    #     164 return _merge_replica_results(replica_results)

    # File /usr/local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py:161, in <genexpr>(.0)
    #     154   mp_fn = functools.partial(
    #     155       _run_thread_per_device,
    #     156       local_world_size=num_processes,
    #     157       fn=functools.partial(fn, *args, **kwargs),
    #     158       initializer_fn=initialize_multiprocess)
    #     159   process_results = executor.map(mp_fn, range(num_processes))
    #     160   replica_results = list(
    # --> 161       itertools.chain.from_iterable(
    #     162           result.items() for result in process_results))
    #     164 return _merge_replica_results(replica_results)

    # File /usr/local/lib/python3.10/concurrent/futures/process.py:575, in _chain_from_iterable_of_lists(iterable)
    #     569 def _chain_from_iterable_of_lists(iterable):
    #     570     """
    #     571     Specialized implementation of itertools.chain.from_iterable.
    #     572     Each item in *iterable* should be a list.  This function is
    #     573     careful not to keep references to yielded objects.
    #     574     """
    # --> 575     for element in iterable:
    #     576         element.reverse()
    #     577         while element:

    # File /usr/local/lib/python3.10/concurrent/futures/_base.py:621, in Executor.map.<locals>.result_iterator()
    #     618 while fs:
    #     619     # Careful not to keep a reference to the popped future
    #     620     if timeout is None:
    # --> 621         yield _result_or_cancel(fs.pop())
    #     622     else:
    #     623         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

    # File /usr/local/lib/python3.10/concurrent/futures/_base.py:319, in _result_or_cancel(***failed resolving arguments***)
    #     317 try:
    #     318     try:
    # --> 319         return fut.result(timeout)
    #     320     finally:
    #     321         fut.cancel()

    # File /usr/local/lib/python3.10/concurrent/futures/_base.py:458, in Future.result(self, timeout)
    #     456     raise CancelledError()
    #     457 elif self._state == FINISHED:
    # --> 458     return self.__get_result()
    #     459 else:
    #     460     raise TimeoutError()

    # File /usr/local/lib/python3.10/concurrent/futures/_base.py:403, in Future.__get_result(self)
    #     401 if self._exception:
    #     402     try:
    # --> 403         raise self._exception
    #     404     finally:
    #     405         # Break a reference cycle with the exception in self._exception
    #     406         self = None

    # BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

    # 训练核心数
    # P100 = 1
    # T4x2 = 2
    # TPU  = 8
    # CPU  = 0
    num_processes = match_num_processes()

    lock = mp.Manager().Lock()

    if num_processes == 8:
        xmp.spawn(run_fn, args=(lock, num_processes, test), start_method='fork')      
    else:
        notebook_launcher(run_fn, args=(0, lock, num_processes, copy.deepcopy(test)), num_processes=num_processes)