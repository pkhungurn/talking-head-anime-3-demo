from dataclasses import dataclass
from io import BytesIO
import logging
from contextlib import contextmanager
import time

from PIL import Image
import numpy as np
import torch
from flask import Flask, send_file, request
from flask_cors import CORS

from tha3.util import (
    extract_pytorch_image_from_PIL_image,
    convert_output_image_from_torch_to_numpy,
    convert_output_image_from_torch_to_pil,
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

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

base_image = Image.open("data/images/aiv_gin.png")
torch_base_image = extract_pytorch_image_from_PIL_image(base_image).to(device)
if poser.get_dtype() == torch.half:
    torch_base_image = torch_base_image.half()

CORS(
    app, resources={r"/*": {"origins": "http://127.0.0.1:8080"}},
)


@contextmanager
def time_estimate(label: str):
    start = time.time()
    try:
        yield
    finally:
        end = time.time() - start
        app.logger.info(f"{label} end: {end} sec")


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


def pil_image_from_pytorch_image(pytorch_image):
    output_image = pytorch_image.detach().cpu()
    np_image = np.uint8(
        np.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0)
    )
    pil_image = Image.fromarray(np_image, mode="RGBA")
    return pil_image


def io_from_pil_image(input_image: Image):
    img_io = BytesIO()
    input_image.save(img_io, "PNG", quality=95)
    img_io.seek(0)
    return img_io


@app.route("/", methods=["GET"])
def get_base_image():
    """
    基本となるイメージを返す
    """
    img_io = io_from_pil_image(base_image)
    res = send_file(img_io, mimetype="image/png")
    # res.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:8080"
    return res


@app.route("/execution", methods=["POST"])
def inference():
    """
    基本となるイメージを返す

    Requests:
        {
            "params": {
                "eyebrow_dropdown": chose in ("troubled", "angry", "lowered", "raised", "happy", "serious"),
                "eyebrow_left": (0.0, 1.0),
                "eyebrow_right":,
                "eye_dropdown":,
                "eye_left":,
                "eye_right":,
                "mouth_dropdown":,
                "mouth_left":,
                "mouth_right":,
                "iris_small_left":,
                "iris_small_right":,
                "iris_rotation_x":,
                "iris_rotation_y":,
                "head_x":,
                "head_y":,
                "neck_z":,
                "body_y":,
                "body_z":,
                "breathing":,
            }
        }
    """
    req = request.json
    # app.logger.info(req)

    input_pose = InputPose(**req["params"])
    with time_estimate("get_pose"):
        pose = input_pose.get_pose()
    with torch.inference_mode():
        with time_estimate("poser.pose"):
            output_image = poser.pose(torch_base_image, pose)[0]
        with time_estimate("pil_image_from_pytorch_image"):
            # image = pil_image_from_pytorch_image(output_image)
            image = convert_output_image_from_torch_to_pil(output_image)

    img_io = io_from_pil_image(image)
    res = send_file(img_io, mimetype="image/png")
    res.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:8080"
    return res
