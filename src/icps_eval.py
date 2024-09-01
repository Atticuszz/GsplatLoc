import json
import sys
from pathlib import Path

from src.eval.experiment import ICPExperiment, RegistrationConfig, WandbConfig
from src.eval.utils import set_random_seed

sys.path.append("..")
set_random_seed(42)


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
        # "GICP",
        # "PLANE_ICP",
        # "COLORED_ICP",
        # "ICP",
        "HYBRID",
    ]
    implements = ("open3d",)
    # rooms = [
    #     # "freiburg1_desk",
    #     # "freiburg1_desk2",
    #     # "freiburg1_room",
    #     # "freiburg2_xyz",
    #     # "freiburg3_long_office_household",
    # ]
    # scenes = "TUM"
    scenes = "Replica"
    # rooms = ["room" + str(i) for i in range(3)]

    # rooms = ["office" + str(i) for i in range(0, 3)]
    rooms = ["office" + str(i) for i in range(3, 4)]
    # rooms = ["office" + str(i) for i in range(3, 5)]

    finished = load_finished_experiments(file_path)

    for room in rooms:
        for method in methods:
            for imp in implements:
                # skip finished
                config_tuple = (room, method, imp)
                if config_tuple in finished:
                    continue
                # small_gicp does not have  color icp
                # if imp == "small_gicp" and method == "COLORED_ICP":
                #     continue
                registration_config = RegistrationConfig(
                    registration_type=method,
                    voxel_downsampling_resolutions=0.0,
                    implementation=imp,
                    # knn=20,
                )
                experiment = ICPExperiment(
                    registration_config=registration_config,
                    wandb_config=WandbConfig(
                        method,
                        dataset=scenes,
                        sub_set=room,
                        implementation=imp,
                        description="icps_v3_test",
                    ),
                )
                try:
                    experiment.run()
                except Exception as e:
                    experiment.logger.finish()
                    print(f"experiment.run() with exception: {e}!")
                save_finished_experiment(file_path, config_tuple)
