import itertools

class skip_first_batches:
    def __init__(self, dataloader, num_batches):
        self.num_batches = num_batches
        self.it = iter(dataloader)
        next(itertools.islice(self.it, num_batches, num_batches), None)  # 跳过前 n 个元素

    def __iter__(self):
        return self 

    def __next__(self):
        return next(self.it)

    def __len__(self):  
        return len(self.it) - self.num_batches

# def skip_first_batches(dataloader, num_batches):
#     it = iter(dataloader)
#     next(itertools.islice(it, num_batches, num_batches), None)  # 跳过前 n 个元素
#     return it