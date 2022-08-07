import math

from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, \
    RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_QUAT

IFACIALMOCAP_PORT = 49983
IFACIALMOCAP_START_STRING = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719|sendDataVersion=v2".encode('utf-8')


def parse_ifacialmocap_v2_pose(ifacialmocap_output):
    output = {}
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if "&" in part:
            components = part.split("&")
            assert len(components) == 2
            key = components[0]
            value = float(components[1]) / 100.0
            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
            if key in BLENDSHAPE_NAMES:
                output[key] = value
        elif part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
            output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
            output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
            output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("rightEye#"):
            components = part[len("rightEye#"):].split(",")
            output[RIGHT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[RIGHT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[RIGHT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("leftEye#"):
            components = part[len("leftEye#"):].split(",")
            output[LEFT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[LEFT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[LEFT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return output


def parse_ifacialmocap_v1_pose(ifacialmocap_output):
    output = {}
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
            output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
            output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
            output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("rightEye#"):
            components = part[len("rightEye#"):].split(",")
            output[RIGHT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[RIGHT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[RIGHT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("leftEye#"):
            components = part[len("leftEye#"):].split(",")
            output[LEFT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[LEFT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[LEFT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        else:
            components = part.split("-")
            assert len(components) == 2
            key = components[0]
            value = float(components[1]) / 100.0
            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
            if key in BLENDSHAPE_NAMES:
                output[key] = value
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return output

