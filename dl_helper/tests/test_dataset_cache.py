from torch.utils.data import DataLoader
import multiprocessing as mp

from dl_helper.trainer import notebook_launcher
from dl_helper.data import DistributedSampler, Dataset_cahce
from dl_helper.train_param import Params
from accelerate import Accelerator

def pprint(lock, *args):
    with lock:
        print(*args)

def test_fn(lock, _type='cache'):
    acc = Accelerator()
    device = acc.device

    dataset_name = 'pred_10@20@30_pass_100_y_3_bd_2024_05_01_dr_4@1@1_th_12_s_ETHFDUSD@ETHUSDT@BTCFDUSD@BTCUSDT_t_10_target_mid_std_5d.7z'

    param = Params(
        'test',
        'test',
        dataset_name,
        0.001,64,data_folder='/kaggle/input/lh-q-bin-data-20240629'
    )

    dataset = Dataset_cahce(param, 'train')

    if _type == 'cache':
        sampler = DistributedSampler(dataset, acc, shuffle=True)
        dataloader = DataLoader(
            dataset,
            64, False, sampler=sampler
        )

        for epoch in param.epochs:
            count = 0
            for mini in sampler.mini_epoch:
                pprint(f'mini_epoch {mini}')
                for data in dataloader:
                    count += 1
            pprint(lock, f'epoch {epoch} count {count}')

    else:
        dataloader = DataLoader(
            dataset,
            64, False
        )

        for epoch in param.epochs:
            count = 0
            for data in dataloader:
                count += 1
            pprint(lock, f'epoch {epoch} count {count}')


def run():
    lock = mp.Lock()

    for _type in ['cache', 'normal']:
        print('-------------------', _type, '---------------------')
        notebook_launcher(test_fn, (lock, _type), num_processes=2)
        print('-------------------', _type, '---------------------')


