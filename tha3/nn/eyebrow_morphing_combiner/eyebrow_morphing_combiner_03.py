from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module

from tha3.nn.common.poser_encoder_decoder_00 import PoserEncoderDecoder00Args
from tha3.nn.common.poser_encoder_decoder_00_separable import PoserEncoderDecoder00Separable
from tha3.nn.image_processing_util import apply_color_change, apply_rgb_change, GridChangeApplier
from tha3.module.module_factory import ModuleFactory
from tha3.nn.nonlinearity_factory import ReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class EyebrowMorphingCombiner03Args(PoserEncoderDecoder00Args):
    def __init__(self,
                 image_size: int = 128,
                 image_channels: int = 4,
                 num_pose_params: int = 12,
                 start_channels: int = 64,
                 bottleneck_image_size=16,
                 num_bottleneck_blocks=6,
                 max_channels: int = 512,
                 block_args: Optional[BlockArgs] = None):
        super().__init__(
            image_size,
            2 * image_channels,
            image_channels,
            num_pose_params,
            start_channels,
            bottleneck_image_size,
            num_bottleneck_blocks,
            max_channels,
            block_args)


class EyebrowMorphingCombiner03(Module):
    def __init__(self, args: EyebrowMorphingCombiner03Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00Separable(args)
        self.morphed_eyebrow_layer_grid_change = self.args.create_grid_change_block()
        self.morphed_eyebrow_layer_alpha = self.args.create_alpha_block()
        self.morphed_eyebrow_layer_color_change = self.args.create_color_change_block()
        self.combine_alpha = self.args.create_alpha_block()
        self.grid_change_applier = GridChangeApplier()

    def forward(self, background_layer: Tensor, eyebrow_layer: Tensor, pose: Tensor, *args) -> List[Tensor]:
        combined_image = torch.cat([background_layer, eyebrow_layer], dim=1)
        feature = self.body(combined_image, pose)[0]

        morphed_eyebrow_layer_grid_change = self.morphed_eyebrow_layer_grid_change(feature)
        morphed_eyebrow_layer_alpha = self.morphed_eyebrow_layer_alpha(feature)
        morphed_eyebrow_layer_color_change = self.morphed_eyebrow_layer_color_change(feature)
        warped_eyebrow_layer = self.grid_change_applier.apply(morphed_eyebrow_layer_grid_change, eyebrow_layer)
        morphed_eyebrow_layer = apply_color_change(
            morphed_eyebrow_layer_alpha, morphed_eyebrow_layer_color_change, warped_eyebrow_layer)

        combine_alpha = self.combine_alpha(feature)
        eyebrow_image = apply_rgb_change(combine_alpha, morphed_eyebrow_layer, background_layer)
        eyebrow_image_no_combine_alpha = apply_rgb_change(
            (morphed_eyebrow_layer[:, 3:4, :, :] + 1.0) / 2.0, morphed_eyebrow_layer, background_layer)

        return [
            eyebrow_image,  # 0
            combine_alpha,  # 1
            eyebrow_image_no_combine_alpha,  # 2
            morphed_eyebrow_layer,  # 3
            morphed_eyebrow_layer_alpha,  # 4
            morphed_eyebrow_layer_color_change,  # 5
            warped_eyebrow_layer,  # 6
            morphed_eyebrow_layer_grid_change,  # 7
        ]

    EYEBROW_IMAGE_INDEX = 0
    COMBINE_ALPHA_INDEX = 1
    EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX = 2
    MORPHED_EYEBROW_LAYER_INDEX = 3
    MORPHED_EYEBROW_LAYER_ALPHA_INDEX = 4
    MORPHED_EYEBROW_LAYER_COLOR_CHANGE_INDEX = 5
    WARPED_EYEBROW_LAYER_INDEX = 6
    MORPHED_EYEBROW_LAYER_GRID_CHANGE_INDEX = 7
    OUTPUT_LENGTH = 8


class EyebrowMorphingCombiner03Factory(ModuleFactory):
    def __init__(self, args: EyebrowMorphingCombiner03Args):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return EyebrowMorphingCombiner03(self.args)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    args = EyebrowMorphingCombiner03Args(
        image_size=128,
        image_channels=4,
        num_pose_params=12,
        start_channels=64,
        bottleneck_image_size=16,
        num_bottleneck_blocks=3,
        block_args=BlockArgs(
            initialization_method='xavier',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=ReLUFactory(inplace=True)))
    face_morpher = EyebrowMorphingCombiner03(args).to(cuda)

    background_layer = torch.randn(8, 4, 128, 128, device=cuda)
    eyebrow_layer = torch.randn(8, 4, 128, 128, device=cuda)
    pose = torch.randn(8, 12, device=cuda)
    outputs = face_morpher.forward(background_layer, eyebrow_layer, pose)
    for i in range(len(outputs)):
        print(i, outputs[i].shape)
