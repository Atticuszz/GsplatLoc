import json
from pathlib import Path

from src.component.eval import Experiment, RegistrationConfig


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


def points_left(v_ds, at_least: int = 500):

    left = v_ds * 1200 * 680
    if left < at_least:
        print("v_ds:", v_ds, "g_ds:", g_ds, "points:", left)
    return left < at_least


# TODO: downsample with different backends
if __name__ == "__main__":
    file_path = Path("grip_ds_finished_experiments.json")
    methods = ["GICP"]
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
        0.001,
        0.0001,
    ]
    voxel_downsampling_resolutions.reverse()
    # filter too less
    too_less_points = []
    for v_ds in voxel_downsampling_resolutions:
        if points_left(v_ds):
            too_less_points.append(v_ds)
    # select knn with
    knns = [10, 20, 30, 40, 50]

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
            for v_ds in voxel_downsampling_resolutions:
                for knn in knns:
                    config_tuple = (room, method, v_ds, knn)
                    if config_tuple in finished or v_ds in too_less_points:
                        continue
                    registration_config = RegistrationConfig(
                        registration_type=method,
                        voxel_downsampling_resolutions=v_ds,
                        grid_downsample_resolution=1,
                        knn=knn,
                    )
                    experiment = Experiment(
                        name=room,
                        registration_config=registration_config,
                    )
                    experiment.run()
                    save_finished_experiment(file_path, config_tuple)
