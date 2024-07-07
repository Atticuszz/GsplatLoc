import time

import torch

from my_gsplat.gs_trainer import Runner
from pose_estimation import DEVICE
from src.eval.experiment import WandbConfig

# from my_gsplat.trainer import Runner


def main():
    config = WandbConfig(
        sub_set="room0",
        algorithm="gsplat",
        implementation="pytorch",
        num_iters=2000,
        normalize=True,
    )

    runner = Runner(config, extra_config={"cam_opt": "quat"})
    # runner = Runner(config, extra_config={"cam_opt": "6d"})
    # runner = Runner(config, extra_config={"cam_opt": "6d+"})
    runner.config.adjust_steps()
    if runner.config.ckpt is not None:
        # run eval only
        ckpt = torch.load(runner.ckpt, map_location=DEVICE)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        # runner.run()
        runner.train()

    if not runner.config.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    main()
