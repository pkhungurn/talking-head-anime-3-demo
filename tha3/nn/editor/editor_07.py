from typing import Optional, List

import torch
from matplotlib import pyplot
from torch import Tensor
from torch.nn import Module, Sequential, Tanh, Sigmoid

from tha3.nn.image_processing_util import GridChangeApplier, apply_color_change
from tha3.nn.common.resize_conv_unet import ResizeConvUNet, ResizeConvUNetArgs
from tha3.util import numpy_linear_to_srgb
from tha3.module.module_factory import ModuleFactory
from tha3.nn.conv import create_conv3_from_block_args, create_conv3
from tha3.nn.nonlinearity_factory import ReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class Editor07Args:
    def __init__(self,
                 image_size: int = 512,
                 image_channels: int = 4,
                 num_pose_params: int = 6,
                 start_channels: int = 32,
                 bottleneck_image_size=32,
                 num_bottleneck_blocks=6,
                 max_channels: int = 512,
                 upsampling_mode: str = 'nearest',
                 block_args: Optional[BlockArgs] = None,
                 use_separable_convolution: bool = False):
        if block_args is None:
            block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False))

        self.block_args = block_args
        self.upsampling_mode = upsampling_mode
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        self.start_channels = start_channels
        self.num_pose_params = num_pose_params
        self.image_channels = image_channels
        self.image_size = image_size
        self.use_separable_convolution = use_separable_convolution


class Editor07(Module):
    def __init__(self, args: Editor07Args):
        super().__init__()
        self.args = args

        self.body = ResizeConvUNet(ResizeConvUNetArgs(
            image_size=args.image_size,
            input_channels=2 * args.image_channels + args.num_pose_params + 2,
            start_channels=args.start_channels,
            bottleneck_image_size=args.bottleneck_image_size,
            num_bottleneck_blocks=args.num_bottleneck_blocks,
            max_channels=args.max_channels,
            upsample_mode=args.upsampling_mode,
            block_args=args.block_args,
            use_separable_convolution=args.use_separable_convolution))
        self.color_change_creator = Sequential(
            create_conv3_from_block_args(
                in_channels=self.args.start_channels,
                out_channels=self.args.image_channels,
                bias=True,
                block_args=self.args.block_args),
            Tanh())
        self.alpha_creator = Sequential(
            create_conv3_from_block_args(
                in_channels=self.args.start_channels,
                out_channels=self.args.image_channels,
                bias=True,
                block_args=self.args.block_args),
            Sigmoid())
        self.grid_change_creator = create_conv3(
            in_channels=self.args.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)
        self.grid_change_applier = GridChangeApplier()

    def forward(self,
                input_original_image: Tensor,
                input_warped_image: Tensor,
                input_grid_change: Tensor,
                pose: Tensor,
                *args) -> List[Tensor]:
        n, c = pose.shape
        pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.image_size, self.args.image_size)
        feature = torch.cat([input_original_image, input_warped_image, input_grid_change, pose], dim=1)

        feature = self.body.forward(feature)[-1]
        output_grid_change = input_grid_change + self.grid_change_creator(feature)

        output_color_change = self.color_change_creator(feature)
        output_color_change_alpha = self.alpha_creator(feature)
        output_warped_image = self.grid_change_applier.apply(output_grid_change, input_original_image)
        output_color_changed = apply_color_change(output_color_change_alpha, output_color_change, output_warped_image)

        return [
            output_color_changed,
            output_color_change_alpha,
            output_color_change,
            output_warped_image,
            output_grid_change,
        ]

    COLOR_CHANGED_IMAGE_INDEX = 0
    COLOR_CHANGE_ALPHA_INDEX = 1
    COLOR_CHANGE_IMAGE_INDEX = 2
    WARPED_IMAGE_INDEX = 3
    GRID_CHANGE_INDEX = 4
    OUTPUT_LENGTH = 5


class Editor07Factory(ModuleFactory):
    def __init__(self, args: Editor07Args):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return Editor07(self.args)


def show_image(pytorch_image):
    numpy_image = ((pytorch_image + 1.0) / 2.0).squeeze(0).numpy()
    numpy_image[0:3, :, :] = numpy_linear_to_srgb(numpy_image[0:3, :, :])
    c, h, w = numpy_image.shape
    numpy_image = numpy_image.reshape((c, h * w)).transpose().reshape((h, w, c))
    pyplot.imshow(numpy_image)
    pyplot.show()


if __name__ == "__main__":
    cuda = torch.device('cuda')

    image_size = 512
    image_channels = 4
    num_pose_params = 6
    args = Editor07Args(
        image_size=512,
        image_channels=4,
        start_channels=32,
        num_pose_params=6,
        bottleneck_image_size=32,
        num_bottleneck_blocks=6,
        max_channels=512,
        upsampling_mode='nearest',
        block_args=BlockArgs(
            initialization_method='he',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=ReLUFactory(inplace=False)))
    module = Editor07(args).to(cuda)

    image_count = 1
    input_image = torch.zeros(image_count, 4, image_size, image_size, device=cuda)
    direct_image = torch.zeros(image_count, 4, image_size, image_size, device=cuda)
    warped_image = torch.zeros(image_count, 4, image_size, image_size, device=cuda)
    grid_change = torch.zeros(image_count, 2, image_size, image_size, device=cuda)
    pose = torch.zeros(image_count, num_pose_params, device=cuda)

    repeat = 100
    acc = 0.0
    for i in range(repeat + 2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        module.forward(input_image, warped_image, grid_change, pose)
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            elapsed_time = start.elapsed_time(end)
            print("%d:" % i, elapsed_time)
            acc = acc + elapsed_time

    print("average:", acc / repeat)
