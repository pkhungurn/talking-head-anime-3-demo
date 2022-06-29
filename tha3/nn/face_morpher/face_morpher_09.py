from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Sequential, Sigmoid, Tanh, Module
from torch.nn.functional import affine_grid, grid_sample

from tha3.nn.common.poser_encoder_decoder_00 import PoserEncoderDecoder00Args
from tha3.nn.common.poser_encoder_decoder_00_separable import PoserEncoderDecoder00Separable
from tha3.nn.image_processing_util import GridChangeApplier
from tha3.module.module_factory import ModuleFactory
from tha3.nn.conv import create_conv3_from_block_args, create_conv3
from tha3.nn.nonlinearity_factory import LeakyReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class FaceMorpher09Args(PoserEncoderDecoder00Args):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 num_pose_params: int = 67,
                 start_channels: int = 16,
                 bottleneck_image_size=4,
                 num_bottleneck_blocks=3,
                 max_channels: int = 512,
                 block_args: Optional[BlockArgs] = None):
        super().__init__(
            image_size,
            image_channels,
            image_channels,
            num_pose_params,
            start_channels,
            bottleneck_image_size,
            num_bottleneck_blocks,
            max_channels,
            block_args)


class FaceMorpher09(Module):
    def __init__(self, args: FaceMorpher09Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00Separable(args)

        self.iris_mouth_grid_change = self.create_grid_change_block()
        self.iris_mouth_color_change = self.create_color_change_block()
        self.iris_mouth_alpha = self.create_alpha_block()

        self.eye_color_change = self.create_color_change_block()
        self.eye_alpha = self.create_alpha_block()

        self.grid_change_applier = GridChangeApplier()

    def create_alpha_block(self):
        return Sequential(
            create_conv3(
                in_channels=self.args.start_channels,
                out_channels=1,
                bias=True,
                initialization_method=self.args.block_args.initialization_method,
                use_spectral_norm=False),
            Sigmoid())

    def create_color_change_block(self):
        return Sequential(
            create_conv3_from_block_args(
                in_channels=self.args.start_channels,
                out_channels=self.args.input_image_channels,
                bias=True,
                block_args=self.args.block_args),
            Tanh())

    def create_grid_change_block(self):
        return create_conv3(
            in_channels=self.args.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // (2 ** level))

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Tensor, *args) -> List[Tensor]:
        feature = self.body(image, pose)[0]

        iris_mouth_grid_change = self.iris_mouth_grid_change(feature)
        iris_mouth_image_0 = self.grid_change_applier.apply(iris_mouth_grid_change, image)
        iris_mouth_color_change = self.iris_mouth_color_change(feature)
        iris_mouth_alpha = self.iris_mouth_alpha(feature)
        iris_mouth_image_1 = self.apply_color_change(iris_mouth_alpha, iris_mouth_color_change, iris_mouth_image_0)

        eye_color_change = self.eye_color_change(feature)
        eye_alpha = self.eye_alpha(feature)
        output_image = self.apply_color_change(eye_alpha, eye_color_change, iris_mouth_image_1.detach())

        return [
            output_image,  # 0
            eye_alpha,  # 1
            eye_color_change,  # 2
            iris_mouth_image_1,  # 3
            iris_mouth_alpha,  # 4
            iris_mouth_color_change,  # 5
            iris_mouth_image_0,  # 6
        ]

    OUTPUT_IMAGE_INDEX = 0
    EYE_ALPHA_INDEX = 1
    EYE_COLOR_CHANGE_INDEX = 2
    IRIS_MOUTH_IMAGE_1_INDEX = 3
    IRIS_MOUTH_ALPHA_INDEX = 4
    IRIS_MOUTH_COLOR_CHANGE_INDEX = 5
    IRIS_MOUTh_IMAGE_0_INDEX = 6

    def merge_down(self, top_layer: Tensor, bottom_layer: Tensor):
        top_layer_rgb = top_layer[:, 0:3, :, :]
        top_layer_a = top_layer[:, 3:4, :, :]
        return bottom_layer * (1 - top_layer_a) + torch.cat([top_layer_rgb * top_layer_a, top_layer_a], dim=1)

    def apply_grid_change(self, grid_change, image: Tensor) -> Tensor:
        n, c, h, w = image.shape
        device = grid_change.device
        grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
        grid = base_grid + grid_change
        resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return resampled_image

    def apply_color_change(self, alpha, color_change, image: Tensor) -> Tensor:
        return color_change * alpha + image * (1 - alpha)


class FaceMorpher09Factory(ModuleFactory):
    def __init__(self, args: FaceMorpher09Args):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return FaceMorpher09(self.args)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    args = FaceMorpher09Args(
        image_size=256,
        image_channels=4,
        num_pose_params=12,
        start_channels=64,
        bottleneck_image_size=32,
        num_bottleneck_blocks=6,
        block_args=BlockArgs(
            initialization_method='xavier',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=LeakyReLUFactory(inplace=True, negative_slope=0.2)))
    module = FaceMorpher09(args).to(cuda)

    image = torch.zeros(16, 4, 256, 256, device=cuda)
    pose = torch.zeros(16, 12, device=cuda)

    state_dict = module.state_dict()
    for key in state_dict:
        print(key, state_dict[key].shape)

    if False:
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
                acc += elapsed_time

        print("average:", acc / repeat)