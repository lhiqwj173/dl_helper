import os
import torch
from datetime import datetime

from accelerate.utils import set_seed

class train_base():
    def __init__(self, params, process_index, lock):
        self.params = params
        set_seed(self.params.seed)
        self.save_folder = os.path.join(self.params.root, 'var')

        self.lock = lock
        self.process_index = process_index

        self.device = None
        self.data_loader = []
        self.criterion = None
        self.model = None
        self.scheduler = None
        self.tracker = None

    def check_cache(self, check_cache_exist=True):
        assert (not None is self.device) and (not None is self.data_loader) and (not None is self.criterion) and (not None is self.model) and (not None is self.scheduler) and (not None is self.tracker), 'must prepare trade parts'
        
        # 判断 save_folder 下是否为空
        if check_cache_exist:
            if not os.path.listdir(self.save_folder):
                return False
            return True

    def save(self):
        pass

    def load(self):
        if self.check_cache():
            return

    def get_fake_data(self, num_samples, num_classes, batch_size):
        data = torch.randn(num_samples, 3, 32, 32)
        target = torch.randint(0, num_classes, (num_samples,))
        dataset = torch.utils.data.TensorDataset(data, target)
        
        # for debug
        # for i in range(8):
        #     self.print(i, data[i][0][0][:5])
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader
        
    def get_device(self):
        if None is self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
        
    def get_data_sampler(self, data_set):
        return None

    def init_data_loader(self, data_loader):
        self.data_loader.append(data_loader)
        return data_loader

    def clear_data_loader(self):
        self.data_loader.clear()

    def init_criterion(self, criterion):
        self.criterion = criterion
        return criterion
        
    def init_model(self, model):
        self.model = model.to(self.device)
        return self.model
    
    def init_scheduler(self, scheduler):
        self.scheduler = scheduler
        return scheduler

    def init_tracker(self, tracker):
        self.tracker = tracker
        return tracker

    def step(self, loss, optimizer):
        loss.backward()
        optimizer.step()
    
    def prepare(self, d, t):
        return d.to(self.device), t.to(self.device)
       
    def print_head(self, idx):
        return f'[{datetime.now()}][{idx}]'

    def print(self, *msg, main=True, **kwargs):
        head = self.print_head(self.process_index)
        with self.lock:
            print(head, *msg, **kwargs)
        
    def cal_output_loss(self, model, data, target, criterion):
        if self.is_main_process():
            report_memory_usage('cal_output_loss 0')
        self.wait_for_everyone()

        output = model(data)

        self.wait_for_everyone()
        if self.is_main_process():
            report_memory_usage('cal_output_loss 1')

        loss = criterion(output, target)
        
        self.wait_for_everyone()
        if self.is_main_process():
            report_memory_usage('cal_output_loss 2')

        return output, loss
  
    def is_main_process(self):
        return True
    
    def wait_for_everyone(self):
        return

    def gather_for_metrics(self, *args):
        if len(args) == 1:
            return args[0]
        return args


