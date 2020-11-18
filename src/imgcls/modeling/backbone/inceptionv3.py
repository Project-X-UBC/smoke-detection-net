"""
    InceptionV3 backbone
    Original file from https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/inceptionv3.py
"""

__all__ = [
    'build_inceptionv3_backbone'
]

import torch
import torch.nn as nn
import torch.nn.init as init
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY


class Concurrent(nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.
    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """

    def __init__(self,
                 axis=1,
                 stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.stack:
            out = torch.stack(tuple(out), dim=self.axis)
        else:
            out = torch.cat(tuple(out), dim=self.axis)
        return out


class InceptConv(nn.Module):
    """
    InceptionV3 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(InceptConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels):
    """
    1x1 version of the InceptionV3 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0)


class MaxPoolBranch(nn.Module):
    """
    InceptionV3 specific max pooling branch block.
    """

    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(nn.Module):
    """
    InceptionV3 specific average pooling branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(AvgPoolBranch, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(nn.Module):
    """
    InceptionV3 specific convolutional 1x1 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(Conv1x1Branch, self).__init__()
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    InceptionV3 specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels_list : list of tuple of int
        List of numbers of output out_channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(nn.Module):
    """
    InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels_list : list of tuple of int
        List of numbers of output out_channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeq3x3Branch, self).__init__()
        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = out_channels
        self.conv1x3 = InceptConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1))
        self.conv3x1 = InceptConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0))

    def forward(self, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = torch.cat((y1, y2), dim=1)
        return x


class InceptionAUnit(nn.Module):
    """
    InceptionV3 type Inception-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(InceptionAUnit, self).__init__()
        assert (out_channels > 224)
        pool_out_channels = out_channels - 224

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=64))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(48, 64),
            kernel_size_list=(1, 5),
            strides_list=(1, 1),
            padding_list=(0, 2)))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1)))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=pool_out_channels))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(nn.Module):
    """
    InceptionV3 type Reduction-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ReductionAUnit, self).__init__()
        assert (in_channels == 288)
        assert (out_channels == 768)

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionBUnit(nn.Module):
    """
    InceptionV3 type Inception-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    mid_channels : int
        Number of output out_channels in the 7x7 branches.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(InceptionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 768)

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0))))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3))))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    InceptionV3 type Reduction-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ReductionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 1280)

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 320),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 192, 192),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionCUnit(nn.Module):
    """
    InceptionV3 type Inception-C unit.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(InceptionCUnit, self).__init__()
        assert (out_channels == 2048)

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=320))
        self.branches.add_module("branch2", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,)))
        self.branches.add_module("branch3", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(448, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1)))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionV3 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input out_channels.
    out_channels : int
        Number of output out_channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(InceptInitBlock, self).__init__()
        assert (out_channels == 192)

        self.conv1 = InceptConv(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0)
        self.conv2 = InceptConv(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0)
        self.conv3 = InceptConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)
        self.conv4 = InceptConv(
            in_channels=64,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv5 = InceptConv(
            in_channels=80,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=0)
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        return x


class InceptionV3(Backbone):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    out_channels : list of list of int
        Number of output out_channels for each unit.
    init_block_channels : int
        Number of output out_channels for the initial unit.
    b_mid_channels : list of int
        Number of middle out_channels for each Inception-B unit.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input out_channels.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 out_channels,
                 init_block_channels,
                 b_mid_channels,
                 dropout_rate=0.5,
                 in_channels=3,
                 num_classes=1000):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.stem = InceptInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels)
        in_channels = init_block_channels

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(out_channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                if unit == InceptionBUnit:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=b_mid_channels[j - 1]))
                else:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        out_channels=out_channels))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.linear.weight, std=0.01)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def freeze(self, freeze_at):
        if freeze_at > 0:
            # freeze stem
            for p in self.stem.parameters():
                p.requires_grad = False
        # TODO: implement freezing for rest of network
        return self

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return {"linear": x}


@BACKBONE_REGISTRY.register()
def build_inceptionv3_backbone(cfg, input_shape: ShapeSpec):
    num_classes = cfg.MODEL.CLSNET.NUM_CLASSES
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    init_block_channels = 192
    out_channels = [[256, 288, 288],
                    [768, 768, 768, 768, 768],
                    [1280, 2048, 2048]]
    b_mid_channels = [128, 160, 160, 192]

    model = InceptionV3(
        in_channels=input_shape.channels,
        out_channels=out_channels,
        init_block_channels=init_block_channels,
        b_mid_channels=b_mid_channels,
        num_classes=num_classes
    ).freeze(freeze_at)

    return model

