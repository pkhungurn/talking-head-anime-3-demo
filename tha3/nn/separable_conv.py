from typing import Optional

from torch.nn import Sequential, Conv2d, ConvTranspose2d, Module

from tha3.nn.normalization import NormalizationLayerFactory
from tha3.nn.util import BlockArgs, wrap_conv_or_linear_module


def create_separable_conv3(in_channels: int, out_channels: int,
                           bias: bool = False,
                           initialization_method='he',
                           use_spectral_norm: bool = False) -> Module:
    return Sequential(
        wrap_conv_or_linear_module(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels),
            initialization_method,
            use_spectral_norm),
        wrap_conv_or_linear_module(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            initialization_method,
            use_spectral_norm))


def create_separable_conv7(in_channels: int, out_channels: int,
                           bias: bool = False,
                           initialization_method='he',
                           use_spectral_norm: bool = False) -> Module:
    return Sequential(
        wrap_conv_or_linear_module(
            Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, bias=False, groups=in_channels),
            initialization_method,
            use_spectral_norm),
        wrap_conv_or_linear_module(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            initialization_method,
            use_spectral_norm))


def create_separable_conv3_block(
        in_channels: int, out_channels: int, block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        wrap_conv_or_linear_module(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        wrap_conv_or_linear_module(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(block_args.normalization_layer_factory).create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())


def create_separable_conv7_block(
        in_channels: int, out_channels: int, block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        wrap_conv_or_linear_module(
            Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, bias=False, groups=in_channels),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        wrap_conv_or_linear_module(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(block_args.normalization_layer_factory).create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())


def create_separable_downsample_block(
        in_channels: int, out_channels: int, is_output_1x1: bool, block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    if is_output_1x1:
        return Sequential(
            wrap_conv_or_linear_module(
                Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False, groups=in_channels),
                block_args.initialization_method,
                block_args.use_spectral_norm),
            wrap_conv_or_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                block_args.initialization_method,
                block_args.use_spectral_norm),
            block_args.nonlinearity_factory.create())
    else:
        return Sequential(
            wrap_conv_or_linear_module(
                Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False, groups=in_channels),
                block_args.initialization_method,
                block_args.use_spectral_norm),
            wrap_conv_or_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                block_args.initialization_method,
                block_args.use_spectral_norm),
            NormalizationLayerFactory.resolve_2d(block_args.normalization_layer_factory)
                .create(out_channels, affine=True),
            block_args.nonlinearity_factory.create())


def create_separable_upsample_block(
        in_channels: int, out_channels: int, block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        wrap_conv_or_linear_module(
            ConvTranspose2d(
                in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False, groups=in_channels),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        wrap_conv_or_linear_module(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            block_args.initialization_method,
            block_args.use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(block_args.normalization_layer_factory)
            .create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())
