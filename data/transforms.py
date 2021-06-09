"""
-------------------------------------------------
   File Name:    transforms.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""


def get_transform(new_size=None, grayscale=False, flip=True):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip, Grayscale

    image_transform = [ToTensor()]

    if new_size is not None:
        image_transform.insert(0, Resize(new_size))

    if flip:
        image_transform.insert(0, RandomHorizontalFlip())

    if grayscale:
        image_transform.insert(-2, Grayscale())


    return Compose(image_transform)
