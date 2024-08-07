from torchvision import transforms as T

import torch
from domainbed.datasets.ffcv_transforms import RandomGrayscale, RandomColorJitter
from ffcv.fields.decoders import IntDecoder, CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

basic = T.Compose(
    [
        T.Resize(size=(224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def ffcv_tf(use_amp=True):

    device = torch.device('cuda')
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    aug_image_pipeline = [
        RandomResizedCropRGBImageDecoder(output_size=(224, 224), scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)),
        RandomHorizontalFlip(),
        RandomColorJitter(1.0, 0.3, 0.3, 0.3, 0.3),
        RandomGrayscale(0.1),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16 if use_amp else torch.float32),
        T.Normalize(mean=mean, std=std), # Normalize using image statistics
    ]

    basic_image_pipeline = [
        CenterCropRGBImageDecoder(output_size=(224, 224), ratio=1.0),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16 if use_amp else torch.float32),
        T.Normalize(mean=mean, std=std), # Normalize using image statistics
    ]

    return aug_image_pipeline, basic_image_pipeline, label_pipeline

def ffcv_add_center_crop(transforms):
    if isinstance(transforms[0], (CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder)):
        return transforms
    
    transforms.insert(0, CenterCropRGBImageDecoder(output_size=(224, 224), ratio=1.0))
    return transforms