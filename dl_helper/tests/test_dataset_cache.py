from torch.utils.data import DataLoader
import multiprocessing as mp
import time, os, psutil

from dl_helper.trainer import notebook_launcher
from dl_helper.data import DistributedSampler, Dataset_cahce
from dl_helper.train_param import Params
from accelerate import Accelerator


def pprint(lock, *args):
    with lock:
        print(*args)

def report_memory_usage(lock, msg=''):
    memory_usage = psutil.virtual_memory()
    pprint(lock, f"{msg} 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")


def test_fn(lock, _type='cache'):
    acc = Accelerator()
    device = acc.device

    dataset_name = 'pred_10@20@30_pass_100_y_3_bd_2024_05_01_dr_10@1@1_th_36_s_ETHFDUSD@ETHUSDT@BTCFDUSD@BTCUSDT_t_10_target_mid_std_5d.7z'

    param = Params(
        'test',
        'test',
        dataset_name,
        0.001,64,data_folder='/kaggle/input/lh-q-bin-data-20240629',
        epochs=5
    )

    dataset = Dataset_cahce(param, 'train')

    if _type == 'cache':
        sampler = DistributedSampler(dataset, acc, shuffle=True, mini_dataset_length=5)
        dataloader = DataLoader(
            dataset,
            64, False, sampler=sampler
        )

        for epoch in range(param.epochs):
            count = 0
            for mini in range(sampler.mini_epoch):
                pprint(lock, device, f'mini_epoch {mini}')
                for data in dataloader:
                    count += 1
            pprint(lock, device, f'epoch {epoch} count {count}')

            if acc.is_main_process:
                report_memory_usage(lock, f'epoch {epoch} done')

    else:
        # 手动加载数据
        data_map = dataset._parse_data_map(dataset.files)
        dataset._load_data_map(data_map)

        dataloader = DataLoader(
            dataset,
            64, False
        )

        for epoch in range(param.epochs):
            count = 0
            for data in dataloader:
                count += 1
            pprint(lock, device, f'epoch {epoch} count {count}')

            if acc.is_main_process:
                report_memory_usage(lock, f'epoch {epoch} done')


def run():
    lock = mp.Lock()

    for _type in ['cache', 'normal']:
        print('-------------------', _type, '---------------------')
        t = time.time()
        notebook_launcher(test_fn, (lock, _type), num_processes=1)
        print(f'耗时: {(time.time() - t) / 60:.3f} min')
        print('-------------------', _type, '---------------------')

