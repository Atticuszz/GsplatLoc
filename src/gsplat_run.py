import time

import torch

from my_gsplat.gs_trainer import Runner
from pose_estimation import DEVICE

# from my_gsplat.trainer import Runner


def main():
    runner = Runner()
    runner.adjust_steps()
    if runner.ckpt is not None:
        # run eval only
        ckpt = torch.load(runner.ckpt, map_location=DEVICE)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not runner.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    main()
