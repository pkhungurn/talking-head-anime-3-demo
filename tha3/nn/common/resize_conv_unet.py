from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import ModuleList, Module, Upsample

from tha3.nn.common.conv_block_factory import ConvBlockFactory
from tha3.nn.nonlinearity_factory import ReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class ResizeConvUNetArgs:
    def __init__(self,
                 image_size: int,
                 input_channels: int,
                 start_channels: int,
                 bottleneck_image_size: int,
                 num_bottleneck_blocks: int,
                 max_channels: int,
                 upsample_mode: str = 'bilinear',
                 block_args: Optional[BlockArgs] = None,
                 use_separable_convolution: bool = False):
        if block_args is None:
            block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False))

        self.use_separable_convolution = use_separable_convolution
        self.block_args = block_args
        self.upsample_mode = upsample_mode
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        self.input_channels = input_channels
        self.start_channels = start_channels
        self.image_size = image_size


class ResizeConvUNet(Module):
    def __init__(self, args: ResizeConvUNetArgs):
        super().__init__()
        self.args = args
        conv_block_factory = ConvBlockFactory(args.block_args, args.use_separable_convolution)

        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(conv_block_factory.create_conv3_block(
            self.args.input_channels,
            self.args.start_channels))
        current_channels = self.args.start_channels
        current_size = self.args.image_size

        size_to_channel = {
            current_size: current_channels
        }
        while current_size > self.args.bottleneck_image_size:
            next_size = current_size // 2
            next_channels = min(self.args.max_channels, current_channels * 2)
            self.downsample_blocks.append(conv_block_factory.create_downsample_block(
                current_channels,
                next_channels,
                is_output_1x1=False))
            current_size = next_size
            current_channels = next_channels
            size_to_channel[current_size] = current_channels

        self.bottleneck_blocks = ModuleList()
        for i in range(self.args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(conv_block_factory.create_resnet_block(current_channels, is_1x1=False))

        self.output_image_sizes = [current_size]
        self.output_num_channels = [current_channels]
        self.upsample_blocks = ModuleList()
        while current_size < self.args.image_size:
            next_size = current_size * 2
            next_channels = size_to_channel[next_size]
            self.upsample_blocks.append(conv_block_factory.create_conv3_block(
                current_channels + next_channels,
                next_channels))
            current_size = next_size
            current_channels = next_channels
            self.output_image_sizes.append(current_size)
            self.output_num_channels.append(current_channels)

        if args.upsample_mode == 'nearest':
            align_corners = None
        else:
            align_corners = False
        self.double_resolution = Upsample(scale_factor=2, mode=args.upsample_mode, align_corners=align_corners)

    def forward(self, feature: Tensor) -> List[Tensor]:
        downsampled_features = []
        for block in self.downsample_blocks:
            feature = block(feature)
            downsampled_features.append(feature)

        for block in self.bottleneck_blocks:
            feature = block(feature)

        outputs = [feature]
        for i in range(0, len(self.upsample_blocks)):
            feature = self.double_resolution(feature)
            feature = torch.cat([feature, downsampled_features[-i - 2]], dim=1)
            feature = self.upsample_blocks[i](feature)
            outputs.append(feature)

        return outputs


if __name__ == "__main__":
    device = torch.device('cuda')

    image_size = 512
    image_channels = 4
    num_pose_params = 6
    args = ResizeConvUNetArgs(
        image_size=512,
        input_channels=10,
        start_channels=32,
        bottleneck_image_size=32,
        num_bottleneck_blocks=6,
        max_channels=512,
        upsample_mode='nearest',
        use_separable_convolution=False,
        block_args=BlockArgs(
            initialization_method='he',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=ReLUFactory(inplace=False)))
    module = ResizeConvUNet(args).to(device)

    image_count = 8
    input = torch.zeros(image_count, 10, 512, 512, device=device)
    outputs = module.forward(input)
    for output in outputs:
        print(output.shape)


    if True:
        repeat = 100
        acc = 0.0
        for i in range(repeat + 2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            module.forward(input)
            end.record()
            torch.cuda.synchronize()
            if i >= 2:
                elapsed_time = start.elapsed_time(end)
                print("%d:" % i, elapsed_time)
                acc = acc + elapsed_time

        print("average:", acc / repeat)