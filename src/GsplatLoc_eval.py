import argparse
import sys
import time

from src.eval.experiment import WandbConfig
from src.eval.utils import set_random_seed
from src.my_gsplat.gs_trainer_total import Runner

sys.path.append("..")
set_random_seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run GSplat training on specified rooms for Replica or TUM datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["Replica", "TUM"],
        required=True,
        help="Specify the dataset to use (Replica or TUM)",
    )
    parser.add_argument("--rooms", nargs="+", help="Specify room names manually")
    parser.add_argument(
        "--all", action="store_true", help="Run for all rooms in the specified dataset"
    )
    parser.add_argument(
        "--room-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Specify a range of room numbers (for Replica only)",
    )
    parser.add_argument(
        "--office-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Specify a range of office numbers (for Replica only)",
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
    if args.dataset == "Replica":
        if args.all:
            return ["room" + str(i) for i in range(3)] + [
                "office" + str(i) for i in range(5)
            ]
        elif args.rooms:
            return args.rooms
        elif args.room_range:
            return [
                "room" + str(i)
                for i in range(args.room_range[0], args.room_range[1] + 1)
            ]
        elif args.office_range:
            return [
                "office" + str(i)
                for i in range(args.office_range[0], args.office_range[1] + 1)
            ]
        else:
            return ["room0"]  # Default to room0 if no option is specified
    elif args.dataset == "TUM":
        if args.all:
            return [
                "freiburg1_desk",
                "freiburg1_desk2",
                "freiburg1_room",
                "freiburg2_xyz",
                "freiburg3_long_office_household",
            ]
        elif args.rooms:
            return args.rooms
        else:
            return ["freiburg1_desk"]


def main():
    args = parse_arguments()
    rooms = get_rooms(args)

    for room in rooms:
        config = WandbConfig(
            dataset=args.dataset,
            sub_set=room,
            algorithm="gsplat_v4_filter_knn10-10_test",
            implementation="pytorch",
            num_iters=args.num_iters,
            normalize=True,
        )

        runner = Runner(config)
        runner.cfg.adjust_steps()
        runner.train()

        if not args.disable_viewer:
            print(f"Viewer running for {room}... Ctrl+C to move to the next room.")
            try:
                time.sleep(10)  # Wait for 10 seconds before moving to the next room
            except KeyboardInterrupt:
                print(f"Moving to the next room...")


if __name__ == "__main__":
    main()
