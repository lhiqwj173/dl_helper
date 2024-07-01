import itertools

def skip_first_batches(dataloader, num_batches):
    it = iter(dataloader)
    next(itertools.islice(it, num_batches, num_batches), None)  # 跳过前 n 个元素
    return it