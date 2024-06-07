import json
from pathlib import Path

from src.eval.experiment import DepthLossExperiment, WandbConfig


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


# TODO: downsample with different backends
if __name__ == "__main__":
    file_path = Path("depth_loss_finished_experiments.json")
    methods = ["depth_loss"]
    implements = "pytorch"
    num_iters = 20
    learning_rate = 1e-6
    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]

    finished = load_finished_experiments(file_path)
    for room in rooms:
        for method in methods:
            for imp in implements:
                # skip finished
                config_tuple = (room, method, imp)
                if config_tuple in finished:
                    continue

                experiment = DepthLossExperiment(
                    wandb_config=WandbConfig(
                        method,
                        sub_set=room,
                        num_iters=num_iters,
                        learning_rate=learning_rate,
                        description="depth_loss for pose estimation",
                        implementation="pytorch",
                    ),
                )
                experiment.run()
                save_finished_experiment(file_path, config_tuple)
