from dataclasses import dataclass

from PIL import Image
import numpy as np
import torch

import streamlit as st

from tha3.util import (
    extract_pytorch_image_from_PIL_image,
    convert_output_image_from_torch_to_numpy,
)

from tha3.app.model.model import (
    device,
    poser,
    pose_size,
    pose_parameters,
    iris_small_left_index,
    iris_small_right_index,
    iris_rotation_x_index,
    iris_rotation_y_index,
    head_x_index,
    head_y_index,
    neck_z_index,
    body_y_index,
    body_z_index,
    breathing_index,
)


def image_upload_component():
    st.header("Image uploader")
    uploaded_file = st.file_uploader("Choose a image file")
    torch_input_image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="upload images", use_column_width=True)
        torch_input_image = extract_pytorch_image_from_PIL_image(image).to(device)
        if poser.get_dtype() == torch.half:
            torch_input_image = torch_input_image.half()

    return torch_input_image


@dataclass
class InputPose:
    eyebrow_dropdown: str
    eyebrow_left: float
    eyebrow_right: float
    eye_dropdown: str
    eye_left: float
    eye_right: float
    mouth_dropdown: str
    mouth_left: float
    mouth_right: float
    iris_small_left: float
    iris_small_right: float
    iris_rotation_x: float
    iris_rotation_y: float
    head_x: float
    head_y: float
    neck_z: float
    body_y: float
    body_z: float
    breathing: float

    def get_pose(self):
        pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())

        eyebrow_name = f"eyebrow_{self.eyebrow_dropdown}"
        eyebrow_left_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_left")
        eyebrow_right_index = pose_parameters.get_parameter_index(
            f"{eyebrow_name}_right"
        )
        pose[0, eyebrow_left_index] = self.eyebrow_left
        pose[0, eyebrow_right_index] = self.eyebrow_right

        eye_name = f"eye_{self.eye_dropdown}"
        eye_left_index = pose_parameters.get_parameter_index(f"{eye_name}_left")
        eye_right_index = pose_parameters.get_parameter_index(f"{eye_name}_right")
        pose[0, eye_left_index] = self.eye_left
        pose[0, eye_right_index] = self.eye_right

        mouth_name = f"mouth_{self.mouth_dropdown}"
        if mouth_name in {"mouth_lowered_corner", "mouth_raised_corner"}:
            mouth_left_index = pose_parameters.get_parameter_index(f"{mouth_name}_left")
            mouth_right_index = pose_parameters.get_parameter_index(
                f"{mouth_name}_right"
            )
            pose[0, mouth_left_index] = self.mouth_left
            pose[0, mouth_right_index] = self.mouth_right
        else:
            mouth_index = pose_parameters.get_parameter_index(mouth_name)
            pose[0, mouth_index] = self.mouth_left

        pose[0, iris_small_left_index] = self.iris_small_left
        pose[0, iris_small_right_index] = self.iris_small_right
        pose[0, iris_rotation_x_index] = self.iris_rotation_x
        pose[0, iris_rotation_y_index] = self.iris_rotation_y
        pose[0, head_x_index] = self.head_x
        pose[0, head_y_index] = self.head_y
        pose[0, neck_z_index] = self.neck_z
        pose[0, body_y_index] = self.body_y
        pose[0, body_z_index] = self.body_z
        pose[0, breathing_index] = self.breathing

        return pose.to(device)


def slider_component():
    eyebrow_dropdown = st.selectbox(
        "eyebrow_dropdown",
        ["troubled", "angry", "lowered", "raised", "happy", "serious"],
    )
    eyebrow_left = st.slider("eyebrow_left", 0.0, 1.0, step=0.01)
    eyebrow_right = st.slider("eyebrow_right", 0.0, 1.0, step=0.01)

    eye_dropdown = st.selectbox(
        "eye_dropdown",
        [
            "wink",
            "happy_wink",
            "surprised",
            "relaxed",
            "unimpressed",
            "raised_lower_eyelid",
        ],
    )
    eye_left = st.slider("eye_left", 0.0, 1.0, step=0.01)
    eye_right = st.slider("eye_right", 0.0, 1.0, step=0.01)

    mouth_dropdown = st.selectbox(
        "mouth_dropdown",
        [
            "aaa",
            "iii",
            "uuu",
            "eee",
            "ooo",
            "delta",
            "lowered_corner",
            "raised_corner",
            "smirk",
        ],
    )
    right_disable = False
    if mouth_dropdown in {"lowered_corner", "raised_corner"}:
        right_disable = True

    mouth_left = st.slider("mouth_left", 0.0, 1.0, step=0.01)
    mouth_right = st.slider("mouth_right", 0.0, 1.0, step=0.01, disabled=right_disable)

    iris_small_left = st.slider("iris_small_left", 0.0, 1.0, step=0.01)
    iris_small_right = st.slider("iris_small_right", 0.0, 1.0, step=0.01)
    iris_rotation_x = st.slider("iris_rotation_x", -1.0, 1.0, 0.0, step=0.01)
    iris_rotation_y = st.slider("iris_rotation_y", -1.0, 1.0, 0.0, step=0.01)

    head_x = st.slider("head_x", -1.0, 1.0, 0.0, step=0.01)
    head_y = st.slider("head_y", -1.0, 1.0, 0.0, step=0.01)
    neck_z = st.slider("neck_z", -1.0, 1.0, 0.0, step=0.01)

    body_y = st.slider("body_y", -1.0, 1.0, 0.0, step=0.01)
    body_z = st.slider("body_z", -1.0, 1.0, 0.0, step=0.01)
    breathing = st.slider("body_z", 0.0, 1.0, step=0.01)
    input_pose = InputPose(
        eyebrow_dropdown,
        eyebrow_left,
        eyebrow_right,
        eye_dropdown,
        eye_left,
        eye_right,
        mouth_dropdown,
        mouth_left,
        mouth_right,
        iris_small_left,
        iris_small_right,
        iris_rotation_x,
        iris_rotation_y,
        head_x,
        head_y,
        neck_z,
        body_y,
        body_z,
        breathing,
    )

    return input_pose


def pil_image_from_pytorch_image(pytorch_image):
    output_image = pytorch_image.detach().cpu()
    np_image = np.uint8(
        np.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0)
    )
    pil_image = Image.fromarray(np_image, mode="RGBA")
    return pil_image


def output_component(torch_input_image: torch.Tensor, input_pose: InputPose):
    if torch_input_image is None:
        st.write("No Input Image")
        return

    pose = input_pose.get_pose()
    output_image = poser.pose(torch_input_image, pose)[0]
    if output_image is not None:
        image = pil_image_from_pytorch_image(output_image)
        st.image(image, caption="modified image", use_column_width=True)


def view():
    st.title("Talking Head Anime 3 by Streamlit")

    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        image = image_upload_component()
    with col2:
        input_pose = slider_component()
    with col3:
        output_component(image, input_pose)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    view()
