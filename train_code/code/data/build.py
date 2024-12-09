from data.APT_data import APTDataset
from torch.utils import data
import torch.distributed as dist


def build_data_loader(image_size, split, batch_size, path, num_workers, local_rank, shuffle=True):
    
    dataset = APTDataset(image_size, split, path)

    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build %s dataset" % (split))

    if 'train' in split:
        print(">>>>>>>>>>>>>> DistributedSampler >>>>>>>>>>>>>>>>")
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
    else:
        print(">>>>>>>>>>>>>> SequentialSampler >>>>>>>>>>>>>>>>")
        sampler = data.SequentialSampler(dataset)

    data_loader = data.DataLoader(
        dataset = dataset, 
        sampler= sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return data_loader

