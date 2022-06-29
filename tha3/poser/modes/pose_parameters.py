from tha3.poser.poser import PoseParameters, PoseParameterCategory


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