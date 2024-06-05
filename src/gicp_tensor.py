from src.component.eval import Experiment

# TODO: downsample with different backends
if __name__ == "__main__":

    grid_downsample_resolutions = [1, 2, 4, 8, 10, 12, 16, 20]
    grid_downsample_resolutions.reverse()

    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]
    finished = []
    for room in rooms:
        for grid_downsample_resolution in grid_downsample_resolutions:
            if (
                room,
                grid_downsample_resolution,
            ) in finished:
                continue
            experiment = Experiment(
                name=room,
            )
            experiment.run()
