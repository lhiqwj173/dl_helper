from dl_helper.train_param import match_num_processes, is_colab, is_kaggle
from dl_helper.tracker import Tracker, Tracker_None
from dl_helper.tracker import MODEL_FINAL, MODEL_BEST, MODEL_DUMMY, TEST_FINAL, TEST_BEST, TEST_DUMMY
from dl_helper.tool import report_memory_usage, check_nan, check_gradients, in_windows
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
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler

from py_ext.tool import log, debug, get_log_folder, _get_caller_info, init_logger
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from py_ext.alist import alist

ses = os.environ.get('TG_SESSION')

from accelerate import notebook_launcher
from accelerate import Accelerator, load_checkpoint_in_model
from accelerate.utils import broadcast, InitProcessGroupKwargs
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import (
    set_seed
)

def package_root(accelerator, params):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # # 拷贝 log 文件夹
        # destination_folder = os.path.join(params.root, 'logs')
        # source_folder = get_log_folder()
        # os.makedirs(destination_folder, exist_ok=True)
        # for file in os.listdir(source_folder):
        #     src = os.path.join(source_folder, file)
        #     target = os.path.join(destination_folder, file)
        #     # 覆盖拷贝文件
        #     shutil.copy(src, target)
        # print('copy log folder done')

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

def save_batch_confidence(data_type, params, model, first_batch_data):
    with torch.no_grad():
        data, target = first_batch_data
        # 前向传播，获取模型输出
        output = model(data)
        # 对输出应用softmax，获取概率分布
        probabilities = F.softmax(output, dim=1)
        # 获取正确类别的预测置信度
        # torch.gather 从 probabilities 中提取对应 target 的概率值
        confidence_scores = torch.gather(probabilities, 1, target.unsqueeze(1)).squeeze()
        # 获取模型预测的类别（概率最大的类别）
        predicted_labels = torch.argmax(probabilities, dim=1)
        # 将结果转换为列表形式
        confidence_scores = confidence_scores.detach().cpu().numpy().tolist()
        predicted_labels = predicted_labels.detach().cpu().numpy().tolist()
        true_labels = target.detach().cpu().numpy().tolist()
        # 追加输出到 csv
        file = os.path.join(params.root, f'{data_type}_batch_confidence.csv')
        if not os.path.exists(file):
            # 写入表头
            with open(file, 'w') as f:
                f.write('time,id,confidence,predicted_label,true_label\n')
        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(file, 'a') as f:
            for idx, (confidence, predicted_label, true_label) in enumerate(zip(confidence_scores, predicted_labels, true_labels)):
                f.write(f'{t},{idx},{confidence},{predicted_label},{true_label}\n')

def train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, printer, trans, need_checkpoint=True):
    # 检查是否存在 step 记录
    skip_steps = tracker.step_count
    # printer.print(f'train_fn step_count: {skip_steps}',main=False)

    active_dataloader = train_loader
    # printer.print(f'train_fn active_dataloader: {id(active_dataloader)}',main=False)
    # if skip_steps > 0:
    #     printer.print(f"[{epoch}] skipping train {skip_steps} steps.")
    #     active_dataloader = skip_first_batches(train_loader, skip_steps)

    first_batch_data = None

    model.train()
    for batch in active_dataloader:
        # printer.print(f'batch begin',main=False)

        # 预处理
        data, target = trans(batch, train=True)
        # printer.print(f'batch data shape: {data.shape}, target shape: {target.shape}',main=False)

        if params.classify:
            target = target.long()

        if None is first_batch_data and accelerator.is_local_main_process:
            first_batch_data = (data.clone(), target.clone())
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # printer.print(f'batch loss',main=False)

        if accelerator.is_local_main_process:
            if check_nan(output):
                pickle.dump((data, target, output), open('error_data.pkl', 'wb'))
                wx.send_message(f'训练数据异常 nan/inf')
                wx.send_file('error_data.pkl')
                raise Exception('训练数据异常 nan/inf')

        accelerator.backward(loss)
        # printer.print(f'batch backward',main=False)
        optimizer.step()
        # printer.print(f'batch step',main=False)

        # 追踪器 记录数据
        with torch.no_grad():
            tracker.track('train', output, data, target, loss)
        # printer.print(f'batch track',main=False)

    # 检查最后batch的输出
    printer.print(f"Output mean: {output.mean().item()}, std: {output.std().item()}") 

    # 检查最后一个 batch 的梯度，并记录
    if accelerator.is_local_main_process:
        total_grad_norm = check_gradients(model)
        tracker.record('total_grad_norm', total_grad_norm)
        # printer.print(f'batch check_gradients',main=False)

    # 检查是否存在NaN
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            msg = f"NaN detected in {name}"
            wx.send_message(msg)
            raise Exception(msg)

    # 追踪器，计算必要的数据
    tracker.update()
    # printer.print(f'batch update')

    # 缓存checkpoint
    if need_checkpoint:
        checkpoint(epoch, accelerator, params, printer, False)
    # printer.print(f'batch checkpoint')

    # 固定第一个 train batch 统计正确类别的预测置信度 
    if accelerator.is_local_main_process:
        save_batch_confidence('train', params, model, first_batch_data)

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
    if params.classify:
        target = target.long()
    output = model(data)
    batch_indices = _check_nan(output)

    print(f"[{idx}] batch_indices: {batch_indices}")



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

    first_batch_data = None

    model.eval()
    with torch.no_grad():
        for batch in active_dataloader:
            data, target = trans(batch)
            if params.classify:
                target = target.long()

            if None is first_batch_data and accelerator.is_local_main_process:
                first_batch_data = (data.clone(), target.clone())

            output = model(data)
            loss = criterion(output, target)

            # 追踪器 记录数据
            tracker.track('val', output, data, target, loss)
    
    # debug('val loop done')

    # 追踪器，计算必要的数据
    tracker.update()
    # debug('val_fn done')

    # 固定第一个 val batch 统计正确类别的预测置信度 
    if accelerator.is_local_main_process:
        save_batch_confidence('val', params, model, first_batch_data)

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
        printer.print(f'测试模型: {i}')# TODO

        model.eval()
        with torch.no_grad():
            for batch in test_data:
                data, target = trans(batch)

                if params.classify:
                    target = target.long()

                output = model(data)
                loss = criterion(output, target)

                # 追踪器 记录数据
                tracker.track(test_types[i], output, data, target, loss)

        # 追踪器，计算必要的数据
        # printer.print('update')
        tracker.update()
        printer.print(f'测试模型: {i} done')

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

                        if params.classify:
                            target = target.long()

                        output = model(data)

                        # 追踪器 记录数据
                        tracker.track(run_type, output, data, target, None)

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

def save_model_fn(params, model, accelerator, input_shape=None):
    accelerator.wait_for_everyone()
    accelerator.save_model(model, os.path.join(params.root, MODEL_FINAL))

    # 导出onnx
    if input_shape:
        model = accelerator.unwrap_model(model)
        if accelerator.is_local_main_process:
            onnex_model_save_path = os.path.join(params.root, MODEL_FINAL, f'model.onnx')
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

def run_fn_gpu(lock, num_processes, test_class, args, kwargs, train_param={}, model=None, only_predict=False):

    # 训练实例
    test = test_class(*args, **kwargs)
    try:

        # 训练参数
        params = test.get_param()
        set_seed(params.seed)

        accelerator = Accelerator(mixed_precision=params.amp if params.amp!='no' else 'no', kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))])
        p = printer(lock, accelerator)
        
        # 在root/title中添加 idx
        title_suffix = test.get_title_suffix()
        params.train_title = f'{params.train_title}_IDX{test.idx}' if not title_suffix else f'{params.train_title}_{title_suffix}_IDX{test.idx}'
        params.root = f'{params.root}_IDX{test.idx}' if not title_suffix else f'{params.root}_{title_suffix}_IDX{test.idx}'

        # 初始化日志
        init_logger(params.train_title, home=params.root, timestamp=False)

        # 检查下载训练文件
        if (not params.debug) and accelerator.is_local_main_process and not in_windows():
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
        
            # 总的batch size 没有变化，所以lr 不需要调整
            # if not params.abs_learning_rate:
            #     # 若不存在绝对学习率，需要基于设备调整lr
            #     l = params.learning_rate
            #     params.learning_rate *= num_processes
            #     p.print(f'learning_rate: {l} -> {params.learning_rate}')

        # 临时额外的训练参数
        if train_param:
            for k, v in train_param.items():
                setattr(params, k, v)
                p.print(f'{k}-> {v}')

        assert params is test.get_param(), f'params is not test.para'

        if None is model:
            model = test.get_model()

        if not only_predict:
            train_loader = test.get_data('train')
            val_loader = test.get_data('val')
            p.print(f'data init')

        # 绝对学习率优先
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate, weight_decay=params.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate if not params.abs_learning_rate else params.abs_learning_rate,weight_decay=params.weight_decay)
        optimizer = test.get_optimizer(model)
        scheduler = test.get_lr_scheduler(optimizer)
        criterion = test.get_criterion()

        # # TEST
        # tracker = Tracker_None()
        # 训练跟踪
        tracker = Tracker(model.__class__.__name__, params, accelerator, scheduler, num_processes, p)
        # 新增到 状态 管理
        accelerator.register_for_checkpointing(tracker)
        accelerator.register_for_checkpointing(scheduler)

        # 不需要准备数据
        if not only_predict:
            model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
                model, optimizer, scheduler, train_loader, val_loader
            )
        else:
            model = accelerator.prepare(model)

        p.print(f'prepare done')

        # 数据增强
        trans = test.get_transform(accelerator.device)

        # 读取可能存在的训练数据（继续训练）
        checkpoint_folder = os.path.join(params.root, 'checkpoint')
        resume_from_checkpoint = os.path.exists(checkpoint_folder)
        if resume_from_checkpoint:
            accelerator.print(f"Resumed from checkpoint: {checkpoint_folder}")
            accelerator.load_state(checkpoint_folder)

            # 检查是否需要调整 lr
            if params.learning_rate != scheduler.scheduler.base_lrs[0]:
                p.print(f'change lr: {scheduler.scheduler.base_lrs[0]} -> {params.learning_rate}')
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = params.learning_rate
                scheduler.scheduler.base_lrs = [params.learning_rate] * len(scheduler.scheduler.base_lrs)

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
            p.print(f'train start',main=False)
            for epoch in range(tracker.epoch_count, params.epochs):
                p.print(f'epoch {epoch} tracker.step_in_epoch: {tracker.step_in_epoch}')
                if tracker.step_in_epoch == 0:
                    p.print(f'epoch {epoch} train_fn',main=False)
                    train_fn(epoch, params, model, criterion, optimizer, train_loader, accelerator, tracker, p, trans)

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
                    save_model_fn(params, model, accelerator)

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

        p.print(f'test start')

        # 准备测试数据
        test_loader = test.get_data('test')
        if test_loader:
            test_loader = accelerator.prepare(test_loader)
            # 测试
            test_fn(params, model, test.get_model(), criterion, test_loader, accelerator, tracker, p, trans)

        # 保存模型
        save_model_fn(params, model, accelerator)

        # 绘图
        tracker.save_result()

        # 输出状态到日志
        tracker.print_state()

        # 输出模型预测，用于模型融合
        if params.need_meta_output:
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
    assert mode in ['normal'], f'mode error: {mode}, must be normal'
    num_processes = match_num_processes()
    lock = mp.Manager().Lock()
    notebook_launcher(run_fn_gpu, args=(lock, num_processes, test_class, args, kwargs, train_param, model, True), num_processes=num_processes)

def run(test_class, *args, mode='normal', train_param={}, model=None, **kwargs):
    """
    mode: normal
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
    base_title= f'{test_class.title_base}_{get_gpu_info()}'
    new_kwargs = {}
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('idx='):
                kwargs['idx'] = int(arg.split('=')[1])
            elif arg.startswith('amp='):
                kwargs['amp'] = arg.split('=')[1]
            elif arg in ['findbest_lr', 'find_lr', 'findlr']:
                kwargs['findbest_lr'] = True
            elif arg == 'test':
                kwargs['test'] = True
            elif '=' in arg:
                # 其他参数
                k, v = arg.split('=')
                kwargs[k] = v
                new_kwargs[k] = v

    if 'findbest_lr' in kwargs: base_title+='_findlr'
    if 'amp' in kwargs and kwargs['amp'] in ['fp8', 'fp16', 'bf16']:
        base_title+=f'_{kwargs["amp"]}'
    for k, v in new_kwargs.items():
        base_title += f'_{k}@{v}'
    if 'test' in kwargs and kwargs['test']:
        kwargs['idx'] = 0
    if 'idx' not in kwargs:
        kwargs['idx'] = get_idx(base_title)

    log(f'begin:{base_title} idx: {kwargs["idx"]}')
    kwargs['train_title'] = base_title

    num_processes = match_num_processes()

    try:
        os.environ.pop('TPU_PROCESS_ADDRESSES')
        os.environ.pop('CLOUD_TPU_TASK_ID')
    except:
        pass

    lock = mp.Manager().Lock()

    if mode == 'normal':
        notebook_launcher(run_fn_gpu, args=(lock, num_processes, test_class, args, kwargs, train_param, model), num_processes=num_processes)
    else:
        raise Exception(f'mode error: {mode}, must be normal')