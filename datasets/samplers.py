from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from torch import distributed as dist


def get_sampler(trainset, valset, ddp_enable):
    if ddp_enable:
        trainsampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        trainsampler = RandomSampler(trainset)
    valsampler = SequentialSampler(valset)
    return trainsampler, valsampler