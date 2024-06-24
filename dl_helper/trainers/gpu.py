from dl_helper.trainers.base import train_base
from accelerate import Accelerator


class train_gpu(train_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = Accelerator(mixed_precision=self.params.amp if self.params.amp!='no' else 'no')
        
    def save(self):
        # 检查训练要素是否都以存在
        self.check_cache(check_cache_exist=False)
        self.wait_for_everyone()
        # 缓存训练数据
        self.accelerator.save_state(self.save_folder)

    def load(self):
        if self.check_cache():
            self.accelerator.load_state(self.save_folder)

    def get_device(self):
        if None is self.device:
            self.device = self.accelerator.device
        return self.device
    
    def step(self, loss, optimizer):
        self.accelerator.backward(loss)
        optimizer.step()

    def get_data_sampler(self, data_set):
        return None

    def init_data_loader(self, data_loader):
        data_loader = self.accelerator.prepare(data_loader)
        return data_loader

    def clear_data_loader(self):
        pass
        
    def init_criterion(self, criterion):
        criterion = self.accelerator.prepare(criterion)
        return criterion
    
    def init_model(self, model):
        model = self.accelerator.prepare(model)
        return model

    def init_scheduler(self, scheduler):
        scheduler = self.accelerator.prepare(scheduler)
        return scheduler

    def init_tracker(self, tracker):
        self.accelerator.register_for_checkpointing(tracker)
        return tracker
    
    def prepare(self, d, t):
        return d, t
 
    def print(self, *msg, main=True, **kwargs):
        head = self.print_head(self.process_index)
        with self.lock:
            if main:
                self.accelerator.print(head, *msg, **kwargs)
            else:
                print(head, *msg, **kwargs)
        
    def cal_output_loss(self, model, data, target, criterion):
        if self.params.amp != 'no':
            output = model(data)
            with self.accelerator.autocast():
                loss = criterion(output, target)
            return output, loss
        else:
            return super().cal_output_loss(model, data, target, criterion)

    def is_main_process(self):
        return self.accelerator.is_main_process

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    def gather_for_metrics(self, *args):
        return self.accelerator.gather_for_metrics(args)
