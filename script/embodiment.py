import os
import yaml


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def normalize_embodiment(embodiment):
    if isinstance(embodiment, str):
        return [embodiment]
    if isinstance(embodiment, (list, tuple)):
        return list(embodiment)
    raise ValueError("embodiment must be a string or a list")


def resolve_embodiment_config(args, embodiment_config_path):
    embodiment_type = normalize_embodiment(args.get("embodiment"))

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name):
        robot_file = embodiment_types[embodiment_name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment files for {embodiment_name}")
        return robot_file

    if len(embodiment_type) == 1:
        robot_file = get_embodiment_file(embodiment_type[0])
        robot_config = get_embodiment_config(robot_file)
        is_dual_arm_urdf = bool(robot_config.get("dual_arm", False))

        args["left_robot_file"] = robot_file
        args["left_embodiment_config"] = robot_config
        args["right_robot_file"] = robot_file
        args["right_embodiment_config"] = robot_config
        args["dual_arm_embodied"] = is_dual_arm_urdf
        args["single_arm_embodied"] = not is_dual_arm_urdf
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["single_arm_embodied"] = False
        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["embodiment"] = embodiment_type
    args["embodiment_name"] = (
        str(embodiment_type[0])
        if len(embodiment_type) == 1
        else str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    )
    return args
