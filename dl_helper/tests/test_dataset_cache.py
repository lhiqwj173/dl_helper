from torch.utils.data import DataLoader
import multiprocessing as mp
import time, os, psutil,datetime

from dl_helper.trainer import notebook_launcher
from dl_helper.data import DistributedSampler, Dataset_cahce
from dl_helper.train_param import Params

from accelerate import Accelerator

from py_ext.tool import log, init_logger

init_logger('test', True)

def report_memory_usage(msg=''):
    memory_usage = psutil.virtual_memory()
    log(f"{msg} 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")


def test_fn(_type='cache'):
    acc = Accelerator()
    device = acc.device

    dataset_name = 'pred_10@20@30_pass_100_y_3_bd_2024_05_01_dr_16@1@1_th_36_s_ETHFDUSD@ETHUSDT@BTCFDUSD@BTCUSDT_t_10_target_mid_std_5d.7z'

    param = Params(
        'test',
        'test',
        dataset_name,
        0.001,64,data_folder='/kaggle/input/lh-q-bin-data-20240629',
        epochs=5
    )

    dataset = Dataset_cahce(param, 'train', device)

    if _type == 'cache':
        sampler = DistributedSampler(dataset, acc, shuffle=True, mini_dataset_length=5)
        dataloader = DataLoader(
            dataset,
            64, False, sampler=sampler
        )

        for epoch in range(param.epochs):
            count = 0
            for mini in range(sampler.mini_epoch):
                print_caost_time = True
                t0 = time.time()

                log(device, f'mini_epoch {mini}')
                for data in dataloader:
                    if print_caost_time:
                        print_caost_time = False
                        _t = time.time()
                        acc.print(f'加载数据，耗时: {(_t - t0):.3f} s')
                        t0 = _t
                        
                    count += 1

                acc.wait_for_everyone()
                acc.print(f'数据遍历完毕 耗时: {time.time() - t0:.3f} s')

            log(device, f'epoch {epoch} count {count}')

            if acc.is_main_process:
                report_memory_usage(f'epoch {epoch} done')

    elif _type == 'normal':
        # 手动加载数据
        data_map = dataset._parse_data_map(dataset.files)
        dataset._load_data_map(data_map)

        dataloader = DataLoader(
            dataset,
            64, False
        )
        dataloader = acc.prepare(dataloader)

        for epoch in range(param.epochs):
            print_caost_time = True
            t0 = time.time()
            count = 0
            for data in dataloader:
                if print_caost_time:
                    print_caost_time = False
                    _t = time.time()
                    acc.print(f'加载数据，耗时: {(_t - t0):.3f} s')
                    t0 = _t

                count += 1

            acc.wait_for_everyone()
            acc.print(f'数据遍历完毕 耗时: {time.time() - t0:.3f} s')
            log(device, f'epoch {epoch} count {count}')

            if acc.is_main_process:
                report_memory_usage(f'epoch {epoch} done')
            acc.wait_for_everyone()

    elif _type == 'acc':
        sampler = DistributedSampler(dataset, acc, shuffle=True, mini_dataset_length=5)
        dataloader = DataLoader(
            dataset,
            64, False, sampler=sampler
        )
        dataloader = acc.prepare(dataloader)

        for epoch in range(param.epochs):
            count = 0
            for mini in range(sampler.mini_epoch):
                log(device, f'mini_epoch {mini}')
                for data in dataloader:
                    count += 1
            log(device, f'epoch {epoch} count {count}')

            if acc.is_main_process:
                report_memory_usage(f'epoch {epoch} done')
            acc.wait_for_everyone()


def run(_type):
    # for _type in ['cache', 'normal']:
    log('-------------------', _type, '---------------------')
    t = time.time()
    notebook_launcher(test_fn, (_type, ), num_processes=2)
    log(f'耗时: {(time.time() - t) / 60:.3f} min')
    log('-------------------', _type, '---------------------')


