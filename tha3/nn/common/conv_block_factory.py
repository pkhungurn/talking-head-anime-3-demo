from typing import Optional

from tha3.nn.conv import create_conv7_block_from_block_args, create_conv3_block_from_block_args, \
    create_downsample_block_from_block_args, create_conv3
from tha3.nn.resnet_block import ResnetBlock
from tha3.nn.resnet_block_seperable import ResnetBlockSeparable
from tha3.nn.separable_conv import create_separable_conv7_block, create_separable_conv3_block, \
    create_separable_downsample_block, create_separable_conv3
from tha3.nn.util import BlockArgs


class ConvBlockFactory:
    def __init__(self,
                 block_args: BlockArgs,
                 use_separable_convolution: bool = False):
        self.use_separable_convolution = use_separable_convolution
        self.block_args = block_args

    def create_conv3(self,
                     in_channels: int,
                     out_channels: int,
                     bias: bool,
                     initialization_method: Optional[str] = None):
        if initialization_method is None:
            initialization_method = self.block_args.initialization_method
        if self.use_separable_convolution:
            return create_separable_conv3(
                in_channels, out_channels, bias, initialization_method, self.block_args.use_spectral_norm)
        else:
            return create_conv3(
                in_channels, out_channels, bias, initialization_method, self.block_args.use_spectral_norm)

    def create_conv7_block(self, in_channels: int, out_channels: int):
        if self.use_separable_convolution:
            return create_separable_conv7_block(in_channels, out_channels, self.block_args)
        else:
            return create_conv7_block_from_block_args(in_channels, out_channels, self.block_args)

    def create_conv3_block(self, in_channels: int, out_channels: int):
        if self.use_separable_convolution:
            return create_separable_conv3_block(in_channels, out_channels, self.block_args)
        else:
            return create_conv3_block_from_block_args(in_channels, out_channels, self.block_args)

    def create_downsample_block(self, in_channels: int, out_channels: int, is_output_1x1: bool):
        if self.use_separable_convolution:
            return create_separable_downsample_block(in_channels, out_channels, is_output_1x1, self.block_args)
        else:
            return create_downsample_block_from_block_args(in_channels, out_channels, is_output_1x1)

    def create_resnet_block(self, num_channels: int, is_1x1: bool):
        if self.use_separable_convolution:
            return ResnetBlockSeparable.create(num_channels, is_1x1, block_args=self.block_args)
        else:
            return ResnetBlock.create(num_channels, is_1x1, block_args=self.block_args)