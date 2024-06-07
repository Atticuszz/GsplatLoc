import json
from pathlib import Path

from src.eval.experiment import Experiment, RegistrationConfig, WandbConfig


def load_finished_experiments(file_path: Path):
    """Load the list of finished experiments from the file."""
    if not file_path.exists():
        return []
    with file_path.open("r") as file:
        return [tuple(json.loads(line)) for line in file]


def save_finished_experiment(file_path: Path, finished: tuple):
    """Append a new finished experiment to the file."""
    with file_path.open("a") as file:
        file.write(json.dumps(finished) + "\n")


def points_left(v_ds, at_least: int = 1000):

    left = v_ds * 1200 * 680
    if left < at_least:
        print("v_ds:", v_ds, "points:", left)
    return left < at_least


# TODO: downsample with different backends
if __name__ == "__main__":
    file_path = Path("grip_o3d_finished_experiments.json")
    methods = [
        "GICP",
        "PLANE_ICP",
        "COLORED_ICP",
        "ICP",
    ]
    implements = "open3d", "small_gicp"
    # implements = (
    #     "small_gicp",
    #     "open3d",
    # )
    voxel_downsampling_resolutions = [
        1,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.01,
    ]
    voxel_downsampling_resolutions.reverse()
    # filter too less
    too_less_points = []
    for v_ds in voxel_downsampling_resolutions:
        if points_left(v_ds):
            too_less_points.append(v_ds)
    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]

    # finished = [
    #     (room, method, v_ds, grid_ds)
    #     for room in rooms[:1]
    #     for method in methods[:1]
    #     for v_ds in voxel_downsampling_resolutions[:3]
    #     for grid_ds in grid_downsample_resolutions
    # ]

    finished = load_finished_experiments(file_path)
    for room in rooms:
        for method in methods:
            for imp in implements:
                for v_ds in voxel_downsampling_resolutions:
                    # skip finished
                    config_tuple = (room, method, imp, v_ds)
                    if config_tuple in finished or v_ds in too_less_points:
                        continue
                    # small_gicp does not have  color icp
                    if imp == "small_gicp" and method == "COLORED_ICP":
                        continue
                    registration_config = RegistrationConfig(
                        registration_type=method,
                        voxel_downsampling_resolutions=v_ds,
                        implementation=imp,
                    )
                    experiment = Experiment(
                        registration_config=registration_config,
                        wandb_config=WandbConfig(
                            method,
                            sub_set=room,
                            description="small_gicp and o3d gcip for test voxel dowmsample influence for accuracy",
                        ),
                    )
                    experiment.run()
                    save_finished_experiment(file_path, config_tuple)
