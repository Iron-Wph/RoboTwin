def get_static_camera_type(camera_config, camera_name, default="D435"):
    for camera_info in camera_config.get("static_camera_list", []):
        if camera_info.get("name") == camera_name:
            return camera_info.get("type", default)
    return default


def get_head_camera_type(camera_config):
    return camera_config.get("head_camera_type") or get_static_camera_type(camera_config, "head_camera")


def get_wrist_camera_type(camera_config):
    return camera_config.get("wrist_camera_type", "D435")


def get_collect_head_camera(camera_config):
    return camera_config.get("collect_head_camera", True)


def get_collect_wrist_camera(camera_config):
    return camera_config.get("collect_wrist_camera", True)
