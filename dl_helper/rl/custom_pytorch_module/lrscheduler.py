from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR

class OneCycleLR(_OneCycleLR):
    def __init__(self, optimizer, total_steps, *args, **kwargs):
        super().__init__(optimizer, total_steps, *args, **kwargs)



