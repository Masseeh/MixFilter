from typing import Type, Union, List, Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet
from functools import partial
from domainbed.mixout.mixlinear import MixConv2d
               

class BasicBlockV2(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if base_width != 64:
            raise ValueError("BasicBlock only supports base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv_layer(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckV2(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        width = int(planes * (base_width / 64.0))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(in_channels=inplanes, out_channels=width, kernel_size=1, stride=1, padding=0, dilation=1)
        # self.conv1 = resnet.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv_layer(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv_layer(in_channels=width, out_channels=planes * self.expansion, kernel_size=1, stride=1, padding=0, dilation=1)
        # self.conv3 = resnet.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + identity)

        return out

class ImageNetResNeXt(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlockV2, BottleneckV2]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.groups = groups
        self.base_width = width_per_group

        conv_layer = partial(MixConv2d, groups=groups, p=0.0, drop_mode='filter', activation=False)
        self._conv_layer = conv_layer

        self.conv1 = conv_layer(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV2) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockV2) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlockV2, BottleneckV2]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        conv_layer = self._conv_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_layer(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, padding=0, dilation=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.base_width,
                  previous_dilation, norm_layer, conv_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def resnext18(pretrained=True, **kwargs):
    model = ImageNetResNeXt(
        block=BasicBlockV2,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        **kwargs
    )

    if pretrained:
        ckp = resnet.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        missing_keys, unexpected_keys = model.load_state_dict(ckp, strict=False)
        if unexpected_keys:
            ValueError(f"Unexpected keys: {unexpected_keys}")
        if missing_keys:
            for key in missing_keys:
                if 'target_w' in key or 'target_b' in key:
                    continue
                else:
                    ValueError(f"Missing key: {key}")

    return model

def resnext50(pretrained=True, **kwargs):
    model = ImageNetResNeXt(
        block=BottleneckV2,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        **kwargs
    )

    if pretrained:
        ckp = resnet.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(check_hash=True)
        # ckp = resnet.ResNet50_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        missing_keys, unexpected_keys = model.load_state_dict(ckp, strict=False)
        if unexpected_keys:
            ValueError(f"Unexpected keys: {unexpected_keys}")
        if missing_keys:
            for key in missing_keys:
                if 'target_w' in key or 'target_b' in key:
                    continue
                else:
                    ValueError(f"Missing key: {key}")

    return model