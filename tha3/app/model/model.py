import torch

from tha3.poser.modes.load_poser import load_poser
from tha3.poser.modes.pose_parameters import get_pose_parameters

MODEL_NAME = "standard_float"
DEVICE_NAME = "cuda"
device = torch.device(DEVICE_NAME)

poser = load_poser(MODEL_NAME, DEVICE_NAME)
poser.get_modules()

pose_size = poser.get_num_parameters()
pose_parameters = get_pose_parameters()
iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")
body_y_index = pose_parameters.get_parameter_index("body_y")
body_z_index = pose_parameters.get_parameter_index("body_z")
breathing_index = pose_parameters.get_parameter_index("breathing")
