import json
import math
import time
from timeit import default_timer

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.my_gsplat.datasets.normalize import transform_points

from .datasets.base import Config
from .datasets.dataset import AlignData, Parser
from .model import CameraOptModule, GSModel
from .utils import DEVICE, CustomEncoder, set_random_seed, to_tensor


class Runner(Config):
    """Engine for training and testing."""

    def __init__(self) -> None:
        super().__init__()
        set_random_seed(42)
        # Setup output directories.
        self.make_dir()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{self.result_dir}/tb")

        # load data
        self.parser = Parser()

        # Pose
        self.pose_optimizers = []
        if self.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.parser)).to(DEVICE)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=self.pose_opt_lr * math.sqrt(self.batch_size),
                    weight_decay=self.pose_opt_reg,
                )
            ]

        # if self.pose_noise > 0.0:
        #     self.pose_perturb = CameraOptModule(len(self.trainset)).to(DEVICE)
        #     self.pose_perturb.random_init(self.pose_noise)

        self.app_optimizers = []

        # Losses & Metrics.
        self.init_loss()

    def train(self):

        # Dump self.
        with open(f"{self.res_dir.as_posix()}/self.json", "w") as f:
            json.dump(vars(self), f, cls=CustomEncoder)

        for i, train_data in enumerate([self.parser[0]]):
            # NOTE: train data loop
            train_data: AlignData

            max_steps = self.max_steps
            # models, optimizers and schedulers
            gs_splats = GSModel(train_data)
            print("Model initialized. Number of GS:", len(gs_splats))

            schedulers = [
                # means3d has a learning rate schedule, that end at 0.01 of the initial value
                torch.optim.lr_scheduler.ExponentialLR(
                    gs_splats.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                ),
            ]
            if self.pose_opt:
                # pose optimization has a learning rate schedule
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    )
                )

            # nerf viewer
            self.init_view(gs_splats.viewer_render_fn)
            init_step = 0
            pbar = tqdm.tqdm(range(init_step, max_steps))
            for step in pbar:
                # NOTE: Training loop.
                global_tic = default_timer()

                if not self.disable_viewer:
                    while self.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()
                    tic = default_timer()
                c2w = train_data.tar_c2w.unsqueeze(0)  # [1, 4, 4]
                c2w_gt = train_data.src_c2w.unsqueeze(0)
                Ks = self.parser.K.unsqueeze(0)  # [1, 3, 3]

                pixels = train_data.pixels.unsqueeze(0) / 255.0  # [1, H, W, 3]

                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                image_ids = to_tensor([i], device=DEVICE, dtype=torch.int32)
                height, width = pixels.shape[1:3]

                # sh schedule
                sh_degree_to_use = min(step // self.sh_degree_interval, self.sh_degree)

                # if self.pose_noise:
                #     c2w = self.pose_perturb(c2w, image_ids)

                if self.pose_opt:
                    c2w = self.pose_adjust(c2w, image_ids)

                # apply c2w to src_gs
                if step > 0:
                    transform_matrix = c2w @ torch.linalg.inv(last_c2w)
                    transformed_points = transform_points(
                        transform_matrix.squeeze(0),
                        train_data.points[train_data.tar_nums :],
                    )
                    train_data.points[train_data.tar_nums :, :].copy_(
                        transformed_points
                    )
                last_c2w = c2w.clone()
                gs_splats.means3d = train_data.points

                # NOTE: gs forward
                renders, alphas, info = gs_splats(
                    camtoworlds=c2w,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if self.depth_loss else "RGB",
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                info["means2d"].retain_grad()  # used for running stats

                # loss
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(
                    pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
                )
                loss = l1loss * (1.0 - self.ssim_lambda) + ssimloss * self.ssim_lambda

                loss.backward()

                desc = f"loss={loss.item():.8f}| " f"sh degree={sh_degree_to_use}| "
                # if self.pose_opt and self.pose_noise:
                if self.pose_opt:
                    # monitor the pose error if we inject noise
                    pose_err = F.l1_loss(c2w_gt, c2w)
                    desc += f"pose err={pose_err.item():.6f}| "
                pbar.set_description(desc)

                if self.tb_every > 0 and step % self.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_GS", len(gs_splats), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    if self.tb_save_image:
                        canvas = (
                            torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        )
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    self.writer.flush()

                # update running stats for prunning & growing
                if step < self.refine_stop_iter:
                    gs_splats.update_running_stats(info)

                    if step > self.refine_start_iter and step % self.refine_every == 0:
                        grads = gs_splats.running_stats[
                            "grad2d"
                        ] / gs_splats.running_stats["count"].clamp_min(1)

                        # grow GSs
                        is_grad_high = grads >= self.grow_grad2d
                        is_small = (
                            torch.exp(gs_splats.scales).max(dim=-1).values
                            <= self.grow_scale3d * train_data.scene_scale
                        )
                        is_dupli = is_grad_high & is_small
                        n_dupli = is_dupli.sum().item()
                        gs_splats.refine_duplicate(is_dupli)

                        is_split = is_grad_high & ~is_small
                        is_split = torch.cat(
                            [
                                is_split,
                                # new GSs added by duplication will not be split
                                torch.zeros(n_dupli, device=DEVICE, dtype=torch.bool),
                            ]
                        )
                        n_split = is_split.sum().item()
                        gs_splats.refine_split(is_split)
                        print(
                            f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                            f"Now having {len(gs_splats)} GSs."
                        )

                        # prune GSs
                        is_prune = torch.sigmoid(gs_splats.opacities) < self.prune_opa
                        if step > self.reset_every:
                            # The official code also implements sreen-size pruning but
                            # it's actually not being used due to a bug:
                            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                            is_too_big = (
                                torch.exp(gs_splats.scales).max(dim=-1).values
                                > self.prune_scale3d * self.scene_scale
                            )
                            is_prune = is_prune | is_too_big
                        n_prune = is_prune.sum().item()
                        gs_splats.refine_keep(~is_prune)
                        print(
                            f"Step {step}: {n_prune} GSs pruned. "
                            f"Now having {len(gs_splats)} GSs."
                        )

                        # reset running stats
                        gs_splats.running_stats["grad2d"].zero_()
                        gs_splats.running_stats["count"].zero_()

                    if step % self.reset_every == 0:
                        gs_splats.reset_opa(self.prune_opa * 2.0)

                # Turn Gradients into Sparse Tensor before running optimizer
                if self.sparse_grad:
                    assert self.packed, "Sparse gradients only work with packed mode."
                    gaussian_ids = info["gaussian_ids"]
                    for k in gs_splats.keys():
                        grad = gs_splats[k].grad
                        if grad is None or grad.is_sparse:
                            continue
                        gs_splats[k].grad = torch.sparse_coo_tensor(
                            indices=gaussian_ids[None],  # [1, nnz]
                            values=grad[gaussian_ids],  # [nnz, ...]
                            size=gs_splats[k].size(),  # [N, ...]
                            is_coalesced=len(Ks) == 1,
                        )

                # optimize
                for optimizer in gs_splats.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()

                # save checkpoint
                if step in [i - 1 for i in self.save_steps] or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(gs_splats),
                    }
                    print("Step: ", step, stats)
                    with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                        json.dump(stats, f)
                    torch.save(
                        {
                            "step": step,
                            "splats": gs_splats.state_dict(),
                        },
                        f"{self.ckpt_dir}/ckpt_{step}.pt",
                    )

                if step in [i - 1 for i in self.eval_steps] or step == max_steps - 1:
                    self.eval(gs_splats, c2w, step)

                # viewer
                if not self.disable_viewer:
                    self.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                    # Update the scene.
                    self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, gs_splats: GSModel, c2w: Tensor, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")

        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        Ks = self.parser.K.unsqueeze(0)
        pixels = self.parser[step].pixels.unsqueeze(0) / 255.0
        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()
        tic = time.time()

        colors, _, _ = gs_splats(
            camtoworlds=c2w,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )  # [1, H, W, 3]
        colors = torch.clamp(colors, 0.0, 1.0)
        torch.cuda.synchronize()
        ellipse_time += time.time() - tic

        # write images
        canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
        imageio.imwrite(
            f"{self.render_dir}/val_{step:04d}.png", (canvas * 255).astype(np.uint8)
        )

        pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        metrics["psnr"].append(self.psnr(colors, pixels))
        metrics["ssim"].append(self.ssim(colors, pixels))
        metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(self.parser)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(gs_splats)}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(gs_splats),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()


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
    torch.autograd.set_detect_anomaly(True)
    main()
