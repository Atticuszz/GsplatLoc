import torch
from torch import Tensor, optim as optim

from src.eval.experiment import WandbConfig
from src.eval.logger import WandbLogger
from src.eval.utils import calculate_translation_error, calculate_rotation_error
from src.pose_estimation import DEVICE
from src.pose_estimation.gemoetry import construct_full_pose
from src.pose_estimation.model import PoseEstimationModel
from src.slam_data import RGBDImage, Replica
from src.utils import to_tensor


def train_model_with_adam(
    tar_rgb_d: RGBDImage,
    src_rgb_d: RGBDImage,
    K: Tensor,
    num_iterations=50,
    learning_rate=1e-3,
    logger: WandbLogger | None = None,
) -> tuple[float, Tensor]:
    init_pose = to_tensor(tar_rgb_d.pose, DEVICE, requires_grad=True)
    model = PoseEstimationModel(K, init_pose, DEVICE, logger)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.01, patience=10
    )

    # result
    min_loss = float("inf")
    best_pose = init_pose.clone()
    for i in range(num_iterations):
        optimizer.zero_grad()
        depth_last = to_tensor(tar_rgb_d.depth, DEVICE, requires_grad=True)
        depth_current = to_tensor(src_rgb_d.depth, DEVICE, requires_grad=True)
        loss = model(depth_last, depth_current, i)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            min_loss = loss.item()
            r, t = model.rotation_cur, model.translation_cur
            cur_pose = construct_full_pose(r, t)
            if loss.item() < min_loss:
                best_pose = cur_pose
            # if i % 10 == 0:
            #     print(f"Iteration {i}: Loss {min_loss}")
            scheduler.step(loss)
            # NOTE: collect data
            if logger is not None:
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.log_lr(cur_lr, i)
                eT = calculate_translation_error(
                    cur_pose.detach().cpu().numpy(), tar_rgb_d.pose
                )
                eR = calculate_rotation_error(
                    cur_pose.detach().cpu().numpy(), tar_rgb_d.pose
                )
                logger.log_translation_error(eT, i)
                logger.log_rotation_error(eR, i)
    return min_loss, best_pose


def train_model_with_LBFGS(
    tar_rgb_d: RGBDImage,
    src_rgb_d: RGBDImage,
    K: Tensor,
    num_iterations=20,
    learning_rate=1e-3,
) -> tuple[float, Tensor]:
    init_pose = to_tensor(tar_rgb_d.pose, DEVICE, requires_grad=True)
    model = PoseEstimationModel(K, init_pose, DEVICE)
    model.to(DEVICE)

    # check if all parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require gradients.")

    # 使用 LBFGS 优化器
    optimizer = optim.LBFGS(
        model.parameters(), lr=learning_rate, max_iter=1, history_size=10
    )

    def closure():
        optimizer.zero_grad()
        depth_last = to_tensor(tar_rgb_d.depth, DEVICE)
        depth_current = to_tensor(src_rgb_d.depth, DEVICE)
        loss = model(depth_last, depth_current)
        loss.backward()
        return loss

    min_loss = float("inf")
    best_pose = None

    # 执行优化
    for i in range(num_iterations):
        optimizer.step(closure)
        loss = closure()
        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                r, t = model.rotation_cur.clone(), model.translation_cur.clone()
                best_pose = construct_full_pose(r, t)
            if i % 10 == 0:
                print(f"Iteration {i}: Loss {min_loss}")

    return min_loss, best_pose


def eval():

    data = Replica()
    method = "depth_loss_test"
    num_iters = 50
    learning_rate = 1e-3

    for i in range(900, 910):
        tar_rgb_d, src_rgb_d = data[i - 1], data[i]
        config = WandbConfig(
            sub_set=data.name,
            algorithm=method,
            implementation="pytorch",
            num_iters=num_iters,
            learning_rate=learning_rate,
            optimizer="adam",
        )
        logger = WandbLogger("pose_estimation", config.as_dict())
        _, estimate_pose = train_model_with_adam(
            tar_rgb_d,
            src_rgb_d,
            to_tensor(tar_rgb_d.K, DEVICE, requires_grad=True),
            num_iterations=num_iters,
            learning_rate=learning_rate,
            logger=logger,
        )
        # _, estimate_pose = train_model_with_LBFGS(
        #     tar_rgb_d, src_rgb_d, to_tensor(tar_rgb_d.K, DEVICE, requires_grad=True)
        # )

        eT = calculate_translation_error(
            estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose
        )
        eR = calculate_rotation_error(
            estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose
        )
        print(f"min Translation error: {eT:.8f}")
        print(f"min Rotation error: {eR:.8f}")
        logger.finish()


if __name__ == "__main__":
    eval()
