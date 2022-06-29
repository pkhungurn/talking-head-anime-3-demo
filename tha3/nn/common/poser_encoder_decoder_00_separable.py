import math
from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import ModuleList, Module

from tha3.nn.common.poser_encoder_decoder_00 import PoserEncoderDecoder00Args
from tha3.nn.resnet_block_seperable import ResnetBlockSeparable
from tha3.nn.separable_conv import create_separable_conv3_block, create_separable_downsample_block, \
    create_separable_upsample_block


class PoserEncoderDecoder00Separable(Module):
    def __init__(self, args: PoserEncoderDecoder00Args):
        super().__init__()
        self.args = args

        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1

        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(
            create_separable_conv3_block(
                args.input_image_channels,
                args.start_channels,
                args.block_args))
        current_image_size = args.image_size
        current_num_channels = args.start_channels
        while current_image_size > args.bottleneck_image_size:
            next_image_size = current_image_size // 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.downsample_blocks.append(create_separable_downsample_block(
                in_channels=current_num_channels,
                out_channels=next_num_channels,
                is_output_1x1=False,
                block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        assert len(self.downsample_blocks) == self.num_levels

        self.bottleneck_blocks = ModuleList()
        self.bottleneck_blocks.append(create_separable_conv3_block(
            in_channels=current_num_channels + args.num_pose_params,
            out_channels=current_num_channels,
            block_args=args.block_args))
        for i in range(1, args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(
                ResnetBlockSeparable.create(
                    num_channels=current_num_channels,
                    is1x1=False,
                    block_args=args.block_args))

        self.upsample_blocks = ModuleList()
        while current_image_size < args.image_size:
            next_image_size = current_image_size * 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.upsample_blocks.append(create_separable_upsample_block(
                in_channels=current_num_channels,
                out_channels=next_num_channels,
                block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // (2 ** level))

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Optional[Tensor] = None) -> List[Tensor]:
        if self.args.num_pose_params != 0:
            assert pose is not None
        else:
            assert pose is None
        outputs = []
        feature = image
        outputs.append(feature)
        for block in self.downsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        if pose is not None:
            n, c = pose.shape
            pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.bottleneck_image_size, self.args.bottleneck_image_size)
            feature = torch.cat([feature, pose], dim=1)
        for block in self.bottleneck_blocks:
            feature = block(feature)
            outputs.append(feature)
        for block in self.upsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        outputs.reverse()
        return outputs
