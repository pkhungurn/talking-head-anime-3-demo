import math
from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Upsample

from tha3.nn.common.conv_block_factory import ConvBlockFactory
from tha3.nn.nonlinearity_factory import LeakyReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class ResizeConvEncoderDecoderArgs:
    def __init__(self,
                 image_size: int,
                 input_channels: int,
                 start_channels: int,
                 bottleneck_image_size,
                 num_bottleneck_blocks,
                 max_channels: int,
                 block_args: Optional[BlockArgs] = None,
                 upsample_mode: str = 'bilinear',
                 use_separable_convolution=False):
        self.use_separable_convolution = use_separable_convolution
        self.upsample_mode = upsample_mode
        self.block_args = block_args
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        self.start_channels = start_channels
        self.image_size = image_size
        self.input_channels = input_channels


class ResizeConvEncoderDecoder(Module):
    def __init__(self, args: ResizeConvEncoderDecoderArgs):
        super().__init__()
        self.args = args

        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1

        conv_block_factory = ConvBlockFactory(args.block_args, args.use_separable_convolution)

        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(conv_block_factory.create_conv7_block(args.input_channels, args.start_channels))
        current_image_size = args.image_size
        current_num_channels = args.start_channels
        while current_image_size > args.bottleneck_image_size:
            next_image_size = current_image_size // 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.downsample_blocks.append(conv_block_factory.create_downsample_block(
                in_channels=current_num_channels,
                out_channels=next_num_channels,
                is_output_1x1=False))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        assert len(self.downsample_blocks) == self.num_levels

        self.bottleneck_blocks = ModuleList()
        for i in range(args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(conv_block_factory.create_resnet_block(current_num_channels, is_1x1=False))

        self.output_image_sizes = [current_image_size]
        self.output_num_channels = [current_num_channels]
        self.upsample_blocks = ModuleList()
        if args.upsample_mode == 'nearest':
            align_corners = None
        else:
            align_corners = False
        while current_image_size < args.image_size:
            next_image_size = current_image_size * 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.upsample_blocks.append(
                Sequential(
                    Upsample(scale_factor=2, mode=args.upsample_mode, align_corners=align_corners),
                    conv_block_factory.create_conv3_block(
                        in_channels=current_num_channels, out_channels=next_num_channels)))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
            self.output_image_sizes.append(current_image_size)
            self.output_num_channels.append(current_num_channels)

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // (2 ** level))

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, feature: Tensor) -> List[Tensor]:
        outputs = []
        for block in self.downsample_blocks:
            feature = block(feature)
        for block in self.bottleneck_blocks:
            feature = block(feature)
        outputs.append(feature)
        for block in self.upsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        return outputs


if __name__ == "__main__":
    device = torch.device('cuda')
    args = ResizeConvEncoderDecoderArgs(
        image_size=512,
        input_channels=4 + 6,
        start_channels=32,
        bottleneck_image_size=32,
        num_bottleneck_blocks=6,
        max_channels=512,
        use_separable_convolution=True,
        block_args=BlockArgs(
            initialization_method='he',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=LeakyReLUFactory(inplace=False, negative_slope=0.1)))
    module = ResizeConvEncoderDecoder(args).to(device)
    print(module.output_image_sizes)
    print(module.output_num_channels)

    input = torch.zeros(8, 4 + 6, 512, 512, device=device)
    outputs = module(input)
    for output in outputs:
        print(output.shape)
