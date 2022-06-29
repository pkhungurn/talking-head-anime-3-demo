from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Tanh

from tha3.nn.image_processing_util import GridChangeApplier
from tha3.nn.common.resize_conv_encoder_decoder import ResizeConvEncoderDecoder, ResizeConvEncoderDecoderArgs
from tha3.module.module_factory import ModuleFactory
from tha3.nn.conv import create_conv3_from_block_args, create_conv3
from tha3.nn.nonlinearity_factory import ReLUFactory, LeakyReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class TwoAlgoFaceBodyRotator05Args:
    def __init__(self,
                 image_size: int = 512,
                 image_channels: int = 4,
                 num_pose_params: int = 6,
                 start_channels: int = 32,
                 bottleneck_image_size=32,
                 num_bottleneck_blocks=6,
                 max_channels: int = 512,
                 upsample_mode: str = 'bilinear',
                 block_args: Optional[BlockArgs] = None,
                 use_separable_convolution=False):
        if block_args is None:
            block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False))

        self.use_separable_convolution = use_separable_convolution
        self.upsample_mode = upsample_mode
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        self.start_channels = start_channels
        self.num_pose_params = num_pose_params
        self.image_channels = image_channels
        self.image_size = image_size
        self.block_args = block_args


class TwoAlgoFaceBodyRotator05(Module):
    def __init__(self, args: TwoAlgoFaceBodyRotator05Args):
        super().__init__()
        self.args = args

        self.encoder_decoder = ResizeConvEncoderDecoder(
            ResizeConvEncoderDecoderArgs(
                image_size=args.image_size,
                input_channels=args.image_channels + args.num_pose_params,
                start_channels=args.start_channels,
                bottleneck_image_size=args.bottleneck_image_size,
                num_bottleneck_blocks=args.num_bottleneck_blocks,
                max_channels=args.max_channels,
                block_args=args.block_args,
                upsample_mode=args.upsample_mode,
                use_separable_convolution=args.use_separable_convolution))

        self.direct_creator = Sequential(
            create_conv3_from_block_args(
                in_channels=self.args.start_channels,
                out_channels=self.args.image_channels,
                bias=True,
                block_args=self.args.block_args),
            Tanh())
        self.grid_change_creator = create_conv3(
            in_channels=self.args.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)
        self.grid_change_applier = GridChangeApplier()

    def forward(self, image: Tensor, pose: Tensor, *args) -> List[Tensor]:
        n, c = pose.shape
        pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.image_size, self.args.image_size)
        feature = torch.cat([image, pose], dim=1)

        feature = self.encoder_decoder.forward(feature)[-1]
        grid_change = self.grid_change_creator(feature)
        direct_image = self.direct_creator(feature)
        warped_image = self.grid_change_applier.apply(grid_change, image)

        return [
            direct_image,
            warped_image,
            grid_change]

    DIRECT_IMAGE_INDEX = 0
    WARPED_IMAGE_INDEX = 1
    GRID_CHANGE_INDEX = 2
    OUTPUT_LENGTH = 3


class TwoAlgoFaceBodyRotator05Factory(ModuleFactory):
    def __init__(self, args: TwoAlgoFaceBodyRotator05Args):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return TwoAlgoFaceBodyRotator05(self.args)


if __name__ == "__main__":
    cuda = torch.device('cuda')

    image_size = 256
    image_channels = 4
    num_pose_params = 6
    args = TwoAlgoFaceBodyRotator05Args(
        image_size=256,
        image_channels=4,
        start_channels=64,
        num_pose_params=6,
        bottleneck_image_size=32,
        num_bottleneck_blocks=6,
        max_channels=512,
        upsample_mode='nearest',
        use_separable_convolution=True,
        block_args=BlockArgs(
            initialization_method='he',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=LeakyReLUFactory(inplace=False, negative_slope=0.1)))
    module = TwoAlgoFaceBodyRotator05(args).to(cuda)

    image_count = 1
    image = torch.zeros(image_count, 4, image_size, image_size, device=cuda)
    pose = torch.zeros(image_count, num_pose_params, device=cuda)

    repeat = 100
    acc = 0.0
    for i in range(repeat + 2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        module.forward(image, pose)
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            elapsed_time = start.elapsed_time(end)
            print("%d:" % i, elapsed_time)
            acc = acc + elapsed_time

    print("average:", acc / repeat)
