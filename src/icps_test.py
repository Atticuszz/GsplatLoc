import json
from pathlib import Path

from src.eval.experiment import ICPExperiment, RegistrationConfig, WandbConfig


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


if __name__ == "__main__":

    file_path = Path("grip_o3d_finished_experiments.json")
    methods = [
        "GICP",
        "PLANE_ICP",
        "COLORED_ICP",
        "ICP",
    ]
    implements = (
        "small_gicp",
        "open3d",
    )

    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]

    finished = load_finished_experiments(file_path)

    for room in rooms:
        for method in methods:
            for imp in implements:
                # skip finished
                config_tuple = (room, method, imp)
                if config_tuple in finished:
                    continue
                # small_gicp does not have  color icp
                if imp == "small_gicp" and method == "COLORED_ICP":
                    continue
                registration_config = RegistrationConfig(
                    registration_type=method,
                    voxel_downsampling_resolutions=0.0,
                    # knn=20,
                )
                experiment = ICPExperiment(
                    registration_config=registration_config,
                    wandb_config=WandbConfig(
                        method,
                        sub_set=room,
                        implementation=imp,
                        description="icps test for accuracy",
                    ),
                )
                experiment.run()
                save_finished_experiment(file_path, config_tuple)
