from src.component.eval import Experiment, RegistrationConfig

# TODO: downsample with different backends
if __name__ == "__main__":
    methods = ["GICP", "ICP", "PLANE_ICP"]
    voxel_downsampling_resolutions = [
        1,
        0.8,
        0.6,
        0.4,
        0.2,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001,
        0.0005,
        0.0001,
    ]
    voxel_downsampling_resolutions.reverse()
    grid_downsample_resolutions = [1, 2, 4, 8, 10, 12, 16, 20]
    grid_downsample_resolutions.reverse()

    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]
    finished = []
    for room in rooms:
        for method in methods:
            for voxel_downsampling_resolution in voxel_downsampling_resolutions:
                for grid_downsample_resolution in grid_downsample_resolutions:
                    if (
                        room,
                        grid_downsample_resolution,
                        voxel_downsampling_resolution,
                    ) in finished:
                        continue
                    registration_config = RegistrationConfig(
                        registration_type=method,
                        voxel_downsampling_resolutions=voxel_downsampling_resolution,
                        grid_downsample_resolution=grid_downsample_resolution,
                    )
                    experiment = Experiment(
                        name=room,
                        registration_config=registration_config,
                    )
                    experiment.run()
