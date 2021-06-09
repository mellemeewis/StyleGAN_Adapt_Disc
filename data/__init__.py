"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from data.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset
from data.transforms import get_transform
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip, Pad
from six.moves import urllib


def make_dataset(cfg):
    if cfg.folder:
        Dataset = FoldersDistributedDataset
    else:
        Dataset = FlatDirectoryImageDataset

    if 'mnist' in cfg.img_dir:

        _dataset = Dataset(data_dir=cfg.img_dir, transform=get_transform(new_size=(cfg.resolution, cfg.resolution), grayscale=True, flip=False))

    else:
        _dataset = Dataset(data_dir=cfg.img_dir, transform=get_transform(new_size=(cfg.resolution, cfg.resolution)))

    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    return dl
