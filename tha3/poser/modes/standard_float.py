from enum import Enum
from typing import List, Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate

from tha3.nn.eyebrow_decomposer.eyebrow_decomposer_00 import EyebrowDecomposer00, \
    EyebrowDecomposer00Factory, EyebrowDecomposer00Args
from tha3.nn.eyebrow_morphing_combiner.eyebrow_morphing_combiner_00 import \
    EyebrowMorphingCombiner00Factory, EyebrowMorphingCombiner00Args, EyebrowMorphingCombiner00
from tha3.nn.face_morpher.face_morpher_08 import FaceMorpher08Args, FaceMorpher08Factory
from tha3.poser.general_poser_02 import GeneralPoser02
from tha3.poser.poser import PoseParameterCategory, PoseParameters
from tha3.nn.editor.editor_07 import Editor07, Editor07Args
from tha3.nn.two_algo_body_rotator.two_algo_face_body_rotator_05 import TwoAlgoFaceBodyRotator05, \
    TwoAlgoFaceBodyRotator05Args
from tha3.util import torch_load
from tha3.compute.cached_computation_func import TensorListCachedComputationFunc
from tha3.compute.cached_computation_protocol import CachedComputationProtocol
from tha3.nn.nonlinearity_factory import ReLUFactory, LeakyReLUFactory
from tha3.nn.normalization import InstanceNorm2dFactory
from tha3.nn.util import BlockArgs


class Network(Enum):
    eyebrow_decomposer = 1
    eyebrow_morphing_combiner = 2
    face_morpher = 3
    two_algo_face_body_rotator = 4
    editor = 5

    @property
    def outputs_key(self):
        return f"{self.name}_outputs"


class Branch(Enum):
    face_morphed_half = 1
    face_morphed_full = 2
    all_outputs = 3


NUM_EYEBROW_PARAMS = 12
NUM_FACE_PARAMS = 27
NUM_ROTATION_PARAMS = 6


class FiveStepPoserComputationProtocol(CachedComputationProtocol):
    def __init__(self, eyebrow_morphed_image_index: int):
        super().__init__()
        self.eyebrow_morphed_image_index = eyebrow_morphed_image_index
        self.cached_batch_0 = None
        self.cached_eyebrow_decomposer_output = None

    def compute_func(self) -> TensorListCachedComputationFunc:
        def func(modules: Dict[str, Module],
                 batch: List[Tensor],
                 outputs: Dict[str, List[Tensor]]):
            if self.cached_batch_0 is None:
                new_batch_0 = True
            elif batch[0].shape[0] != self.cached_batch_0.shape[0]:
                new_batch_0 = True
            else:
                new_batch_0 = torch.max((batch[0] - self.cached_batch_0).abs()).item() > 0
            if not new_batch_0:
                outputs[Network.eyebrow_decomposer.outputs_key] = self.cached_eyebrow_decomposer_output
            output = self.get_output(Branch.all_outputs.name, modules, batch, outputs)
            if new_batch_0:
                self.cached_batch_0 = batch[0]
                self.cached_eyebrow_decomposer_output = outputs[Network.eyebrow_decomposer.outputs_key]
            return output

        return func

    def compute_output(self, key: str, modules: Dict[str, Module], batch: List[Tensor],
                       outputs: Dict[str, List[Tensor]]) -> List[Tensor]:
        if key == Network.eyebrow_decomposer.outputs_key:
            input_image = batch[0][:, :, 64:192, 64 + 128:192 + 128]
            return modules[Network.eyebrow_decomposer.name].forward(input_image)
        elif key == Network.eyebrow_morphing_combiner.outputs_key:
            eyebrow_decomposer_output = self.get_output(Network.eyebrow_decomposer.outputs_key, modules, batch, outputs)
            background_layer = eyebrow_decomposer_output[EyebrowDecomposer00.BACKGROUND_LAYER_INDEX]
            eyebrow_layer = eyebrow_decomposer_output[EyebrowDecomposer00.EYEBROW_LAYER_INDEX]
            eyebrow_pose = batch[1][:, :NUM_EYEBROW_PARAMS]
            return modules[Network.eyebrow_morphing_combiner.name].forward(
                background_layer,
                eyebrow_layer,
                eyebrow_pose)
        elif key == Network.face_morpher.outputs_key:
            eyebrow_morphing_combiner_output = self.get_output(
                Network.eyebrow_morphing_combiner.outputs_key, modules, batch, outputs)
            eyebrow_morphed_image = eyebrow_morphing_combiner_output[self.eyebrow_morphed_image_index]
            input_image = batch[0][:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)].clone()
            input_image[:, :, 32:32 + 128, 32:32 + 128] = eyebrow_morphed_image
            face_pose = batch[1][:, NUM_EYEBROW_PARAMS:NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS]
            return modules[Network.face_morpher.name].forward(input_image, face_pose)
        elif key == Branch.face_morphed_full.name:
            face_morpher_output = self.get_output(Network.face_morpher.outputs_key, modules, batch, outputs)
            face_morphed_image = face_morpher_output[0]
            input_image = batch[0].clone()
            input_image[:, :, 32:32 + 192, 32 + 128:32 + 192 + 128] = face_morphed_image
            return [input_image]
        elif key == Branch.face_morphed_half.name:
            face_morphed_full = self.get_output(Branch.face_morphed_full.name, modules, batch, outputs)[0]
            return [
                interpolate(face_morphed_full, size=(256, 256), mode='bilinear', align_corners=False)
            ]
        elif key == Network.two_algo_face_body_rotator.outputs_key:
            face_morphed_half = self.get_output(Branch.face_morphed_half.name, modules, batch, outputs)[0]
            rotation_pose = batch[1][:, NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS:]
            return modules[Network.two_algo_face_body_rotator.name].forward(face_morphed_half, rotation_pose)
        elif key == Network.editor.outputs_key:
            input_original_image = self.get_output(Branch.face_morphed_full.name, modules, batch, outputs)[0]
            rotator_outputs = self.get_output(
                Network.two_algo_face_body_rotator.outputs_key, modules, batch, outputs)
            half_warped_image = rotator_outputs[TwoAlgoFaceBodyRotator05.WARPED_IMAGE_INDEX]
            full_warped_image = interpolate(
                half_warped_image, size=(512, 512), mode='bilinear', align_corners=False)
            half_grid_change = rotator_outputs[TwoAlgoFaceBodyRotator05.GRID_CHANGE_INDEX]
            full_grid_change = interpolate(
                half_grid_change, size=(512, 512), mode='bilinear', align_corners=False)
            rotation_pose = batch[1][:, NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS:]
            return modules[Network.editor.name].forward(
                input_original_image, full_warped_image, full_grid_change, rotation_pose)
        elif key == Branch.all_outputs.name:
            editor_output = self.get_output(Network.editor.outputs_key, modules, batch, outputs)
            rotater_output = self.get_output(Network.two_algo_face_body_rotator.outputs_key, modules, batch, outputs)
            face_morpher_output = self.get_output(Network.face_morpher.outputs_key, modules, batch, outputs)
            eyebrow_morphing_combiner_output = self.get_output(
                Network.eyebrow_morphing_combiner.outputs_key, modules, batch, outputs)
            eyebrow_decomposer_output = self.get_output(
                Network.eyebrow_decomposer.outputs_key, modules, batch, outputs)
            output = editor_output \
                     + rotater_output \
                     + face_morpher_output \
                     + eyebrow_morphing_combiner_output \
                     + eyebrow_decomposer_output
            return output
        else:
            raise RuntimeError("Unsupported key: " + key)


def load_eyebrow_decomposer(file_name: str):
    factory = EyebrowDecomposer00Factory(
        EyebrowDecomposer00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    print("Loading the eyebrow decomposer ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_eyebrow_morphing_combiner(file_name: str):
    factory = EyebrowMorphingCombiner00Factory(
        EyebrowMorphingCombiner00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            num_pose_params=12,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    print("Loading the eyebrow morphing conbiner ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_face_morpher(file_name: str):
    factory = FaceMorpher08Factory(
        FaceMorpher08Args(
            image_size=192,
            image_channels=4,
            num_expression_params=27,
            start_channels=64,
            bottleneck_image_size=24,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False))))
    print("Loading the face morpher ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_two_algo_generator(file_name) -> Module:
    module = TwoAlgoFaceBodyRotator05(
        TwoAlgoFaceBodyRotator05Args(
            image_size=256,
            image_channels=4,
            start_channels=64,
            num_pose_params=6,
            bottleneck_image_size=32,
            num_bottleneck_blocks=6,
            max_channels=512,
            upsample_mode='nearest',
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=LeakyReLUFactory(inplace=False, negative_slope=0.1))))
    print("Loading the face-body rotator ... ", end="")
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_editor(file_name) -> Module:
    module = Editor07(
        Editor07Args(
            image_size=512,
            image_channels=4,
            num_pose_params=6,
            start_channels=32,
            bottleneck_image_size=64,
            num_bottleneck_blocks=6,
            max_channels=512,
            upsampling_mode='nearest',
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=LeakyReLUFactory(inplace=False, negative_slope=0.1))))
    print("Loading the combiner ... ", end="")
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def get_pose_parameters():
    return PoseParameters.Builder() \
        .add_parameter_group("eyebrow_troubled", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_angry", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_lowered", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_raised", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_happy", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_serious", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eye_wink", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_happy_wink", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_surprised", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_relaxed", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_unimpressed", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_raised_lower_eyelid", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("iris_small", PoseParameterCategory.IRIS_MORPH, arity=2) \
        .add_parameter_group("mouth_aaa", PoseParameterCategory.MOUTH, arity=1, default_value=1.0) \
        .add_parameter_group("mouth_iii", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_uuu", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_eee", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_ooo", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_delta", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_lowered_corner", PoseParameterCategory.MOUTH, arity=2) \
        .add_parameter_group("mouth_raised_corner", PoseParameterCategory.MOUTH, arity=2) \
        .add_parameter_group("mouth_smirk", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("iris_rotation_x", PoseParameterCategory.IRIS_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("iris_rotation_y", PoseParameterCategory.IRIS_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("head_x", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("head_y", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("neck_z", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("body_y", PoseParameterCategory.BODY_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("body_z", PoseParameterCategory.BODY_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("breathing", PoseParameterCategory.BREATHING, arity=1, range=(0.0, 1.0)) \
        .build()


def create_poser(
        device: torch.device,
        module_file_names: Optional[Dict[str, str]] = None,
        eyebrow_morphed_image_index: int = EyebrowMorphingCombiner00.EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX,
        default_output_index: int = 0) -> GeneralPoser02:
    if module_file_names is None:
        module_file_names = {}
    if Network.eyebrow_decomposer.name not in module_file_names:
        dir = "data/models/standard_float"
        file_name = dir + "/eyebrow_decomposer.pt"
        module_file_names[Network.eyebrow_decomposer.name] = file_name
    if Network.eyebrow_morphing_combiner.name not in module_file_names:
        dir = "data/models/standard_float"
        file_name = dir + "/eyebrow_morphing_combiner.pt"
        module_file_names[Network.eyebrow_morphing_combiner.name] = file_name
    if Network.face_morpher.name not in module_file_names:
        dir = "data/models/standard_float"
        file_name = dir + "/face_morpher.pt"
        module_file_names[Network.face_morpher.name] = file_name
    if Network.two_algo_face_body_rotator.name not in module_file_names:
        dir = "data/models/standard_float"
        file_name = dir + "/two_algo_face_body_rotator.pt"
        module_file_names[Network.two_algo_face_body_rotator.name] = file_name
    if Network.editor.name not in module_file_names:
        dir = "data/models/standard_float"
        file_name = dir + "/editor.pt"
        module_file_names[Network.editor.name] = file_name

    loaders = {
        Network.eyebrow_decomposer.name:
            lambda: load_eyebrow_decomposer(module_file_names[Network.eyebrow_decomposer.name]),
        Network.eyebrow_morphing_combiner.name:
            lambda: load_eyebrow_morphing_combiner(module_file_names[Network.eyebrow_morphing_combiner.name]),
        Network.face_morpher.name:
            lambda: load_face_morpher(module_file_names[Network.face_morpher.name]),
        Network.two_algo_face_body_rotator.name:
            lambda: load_two_algo_generator(module_file_names[Network.two_algo_face_body_rotator.name]),
        Network.editor.name:
            lambda: load_editor(module_file_names[Network.editor.name]),
    }
    return GeneralPoser02(
        image_size=512,
        module_loaders=loaders,
        pose_parameters=get_pose_parameters().get_pose_parameter_groups(),
        output_list_func=FiveStepPoserComputationProtocol(eyebrow_morphed_image_index).compute_func(),
        subrect=None,
        device=device,
        output_length=29,
        default_output_index=default_output_index)


if __name__ == "__main__":
    device = torch.device('cuda')
    poser = create_poser(device)

    image = torch.zeros(1, 4, 512, 512, device=device)
    pose = torch.zeros(1, 45, device=device)

    repeat = 100
    acc = 0.0
    for i in range(repeat + 2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        poser.pose(image, pose)
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            elapsed_time = start.elapsed_time(end)
            print("%d:" % i, elapsed_time)
            acc = acc + elapsed_time

    print("average:", acc / repeat)
