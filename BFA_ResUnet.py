########################################################################################################################
# BFA_ResUnet
########################################################################################################################

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from .backbone_resnet import resnet50
from .Unet_decode import Up, OutConv
from torch.nn import functional as F

import torch.nn.functional as F
import cv2
import numpy as np


class BFA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BFA, self).__init__()
        # 修改此处以适应输入通道数量的变化
        self.conv1x1 = nn.Conv2d(in_channels + 1, out_channels, kernel_size=1)  # 注意这里是 in_channels + 1
        self.relu = nn.ReLU()

    def forward(self, x):
        # 将输入图像转换为灰度图像
        gray_image = self.to_grayscale(x)

        # 使用Canny算子进行边缘检测
        edge_image = self.canny_edge_detection(gray_image)

        # 将边缘图像与原图像进行通道级联
        concatenated_image = torch.cat([x, edge_image], dim=1)

        # 通过1x1卷积动态调整特征权重
        output = self.conv1x1(concatenated_image)
        output = self.relu(output)

        return output

    def to_grayscale(self, x):
        # 将输入图像转换为灰度图像
        gray_image = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        gray_image = gray_image.unsqueeze(1)  # 添加通道维度
        return gray_image

    def canny_edge_detection(self, gray_image):
        # 将灰度图像转换为numpy数组
        gray_image_np = gray_image.cpu().detach().numpy().astype(np.uint8)

        # 使用Canny算子进行边缘检测
        edge_image_np = np.zeros_like(gray_image_np)
        for i in range(gray_image_np.shape[0]):
            edge_image_np[i, 0] = cv2.Canny(gray_image_np[i, 0], 50, 150)

        # 将边缘图像转换回PyTorch张量
        edge_image = torch.from_numpy(edge_image_np).float().to(gray_image.device)
        return edge_image

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BFA_resunet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(BFA_resunet, self).__init__()
        backbone = resnet50()

        if pretrain_backbone:
            backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        return_layers = {'relu': 'out0', 'layer1': 'out1', 'layer2': 'out2', 'layer3': 'out3', 'layer4': 'out4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # 在这里添加BFA模块，并确保通道数正确
        self.bfa = BFA(in_channels=self.stage_out_channels[4], out_channels=self.stage_out_channels[4])

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])

        self.conv = OutConv(64, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        result = OrderedDict()
        backbone_out = self.backbone(x)

        # 应用BFA模块于最深层的编码层特征图
        backbone_out['out4'] = self.bfa(backbone_out['out4'])

        x = self.up1(backbone_out['out4'], backbone_out['out3'])
        x = self.up2(x, backbone_out['out2'])
        x = self.up3(x, backbone_out['out1'])
        x = self.up4(x, backbone_out['out0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        return result