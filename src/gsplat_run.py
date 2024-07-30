import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1].as_posix()
sys.path.append(ROOT)
from my_gsplat.gs_trainer_total import Runner
from src.eval.experiment import WandbConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run GSplat training on specified rooms."
    )
    parser.add_argument(
        "--rooms",
        nargs="+",
        help="Specify room names manually (e.g., room0 room1 office0)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run for all rooms (room0-2 and office0-4)"
    )
    parser.add_argument(
        "--room-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Specify a range of room numbers (e.g., 0 2 for room0 to room2)",
    )
    parser.add_argument(
        "--office-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Specify a range of office numbers (e.g., 0 4 for office0 to office4)",
    )
    parser.add_argument(
        "--cam-opt",
        choices=["quat", "6d", "6d+"],
        default="quat",
        help="Camera optimization method (default: quat)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=2000,
        help="Number of iterations (default: 2000)",
    )
    parser.add_argument(
        "--disable-viewer",
        action="store_true",
        default=True,
        help="Disable the viewer",
    )
    return parser.parse_args()


def get_rooms(args):
    if args.all:
        return ["room" + str(i) for i in range(3)] + [
            "office" + str(i) for i in range(5)
        ]
    elif args.rooms:
        return args.rooms
    elif args.room_range:
        return [
            "room" + str(i) for i in range(args.room_range[0], args.room_range[1] + 1)
        ]
    elif args.office_range:
        return [
            "office" + str(i)
            for i in range(args.office_range[0], args.office_range[1] + 1)
        ]
    else:
        return ["room0"]  # Default to room0 if no option is specified


def main():
    args = parse_arguments()
    rooms = get_rooms(args)

    for room in rooms:
        config = WandbConfig(
            sub_set=room,
            algorithm="gsplat_v4_filter_knn10-10",
            implementation="pytorch",
            num_iters=args.num_iters,
            normalize=True,
        )

        runner = Runner(config, extra_config={"cam_opt": args.cam_opt})
        runner.config.adjust_steps()
        runner.train()

        if not args.disable_viewer:
            print(f"Viewer running for {room}... Ctrl+C to move to the next room.")
            try:
                time.sleep(10)  # Wait for 10 seconds before moving to the next room
            except KeyboardInterrupt:
                print(f"Moving to the next room...")


if __name__ == "__main__":
    main()
