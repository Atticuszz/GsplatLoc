import time

from my_gsplat.gs_trainer_total import Runner
from src.eval.experiment import WandbConfig

# from my_gsplat.gs_trainer_total2 import Runner


# from my_gsplat.gs_trainer import Runner


# from my_gsplat.trainer import Runner


def main():
    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]
    for room in rooms:
        config = WandbConfig(
            # sub_set="office2",
            sub_set=room,
            algorithm="gsplat_v3_filter_knn10-10",
            # algorithm="gsplat_outlier",
            implementation="pytorch",
            num_iters=2000,
            normalize=True,
        )

        # runner = Runner(config, extra_config={"cam_opt": "quat"})
        runner = Runner(config, extra_config={"cam_opt": "6d"})
        # runner = Runner(config, extra_config={"cam_opt": "6d+"})
        runner.config.adjust_steps()
        runner.train()

        if not runner.config.disable_viewer:
            print("Viewer running... Ctrl+C to exit.")
            time.sleep(1000000)


if __name__ == "__main__":
    main()
