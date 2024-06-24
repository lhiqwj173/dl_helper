from dl_helper.train_param import match_num_processes
from dl_helper.trainers.base import train_base

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

class train_tpu(train_base):
    def __init__(self, *args, **kwargs):
        dist.init_process_group('xla', init_method='xla://')
        super().__init__(*args, **kwargs)

    def _get_cache_path(self, item, name):
        return os.path.join(self.save_folder, f'{item.__class__.__name__}.pt' if not name else f'{name}.pt')

    def _save(self, item, name=''):
        xm.save(item.state_dict(), _get_cache_path(item, name))

    def _load(self, item, name=''):
        state_dict = xm.load(_get_cache_path(item, name))
        item.load_state_dict(state_dict)

    def save(self):
        # 检查训练要素是否都以存在
        self.check_cache(check_cache_exist=False)
        self.wait_for_everyone()
        # 缓存训练数据
        for i in [
            self.model,
            self.criterion,
            self.scheduler,
            self.tracker
        ]:
            self._save(i)
        for idx, i in enumerate(self.data_loader):
            self._save(i, f'data_loader_{idx}')
        
    def load(self):
        if self.check_cache():
            for i in [
                self.model,
                self.criterion,
                self.scheduler,
                self.tracker
            ]:
                self._load(i)
            for idx, i in enumerate(self.data_loader):
                self._load(i, f'data_loader_{idx}')

    def get_data_sampler(self, data_set):
        train_sampler = None
        if xm.xrt_world_size() > 1:
          train_sampler = torch.utils.data.distributed.DistributedSampler(
              data_set,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=True)

        return train_sampler

    def init_data_loader(self, data_loader):
        train_device_loader = pl.MpDeviceLoader(data_loader, self.device)
        self.data_loader.append(train_device_loader)
        return train_device_loader

    def clear_data_loader(self):
        self.data_loader.clear()

    def get_device(self):
        if None is self.device:
            self.device = xm.xla_device()
        return self.device
    
    def step(self, loss, optimizer):
        loss.backward()
        optimizer.step()

        # ddp模式，不需要 xm.optimizer_step，会自动同步梯度
        # xm.optimizer_step(optimizer)
        
        # for debug
        # 汇总loss
        # _loss = loss.item()
        # print(_loss)
        # xm.mark_step()
        # losss = xm.all_gather(_loss)
        # print(_loss)
    
    def init_model(self, model):
        # self.print(f'init model {self.device}')
        model = model.to(self.device)
        # Initialization is nondeterministic with multiple threads in PjRt.
        # Synchronize model parameters across replicas manually.
        if xr.using_pjrt():
            xm.broadcast_master_param(model)
        self.model = DDP(model, gradient_as_bucket_view=True)
        return self.model
    
    def prepare(self, d, t):
        return d, t
    
    def print(self, *msg, main=True, **kwargs):
        head = self.print_head(self.process_index)
        with self.lock:
            if main:
                if self.is_main_process():
                    print(head, *msg, **kwargs)

                # xm.master_print(*msg)
            else:
                print(head, *msg, **kwargs)

        
    def cal_output_loss(self, model, data, target, criterion):
        if self.params.amp != 'no':
            with autocast(xm.xla_device()):
                output = model(data)
                loss = criterion(output, target)
            return output, loss
        
        else:
            return super().cal_output_loss(model, data, target, criterion)

    def is_main_process(self):
        return xm.is_master_ordinal()

    def wait_for_everyone(self):
        xm.mark_step()

    def gather_for_metrics(self, *args):
        res = [xm.all_gather(i) for i in args]
        if len(res) == 1:
            return res[0]
        return res
