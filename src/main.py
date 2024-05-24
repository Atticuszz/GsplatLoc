"""
-*- coding: utf-8 -*-
@Organization : SupaVision
@Author       : 18317
@Date Created : 03/02/2024
@Description  :
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from slam import Slam2D

LOG_FILE = Path(__file__).parents[1] / "tests/main.log"


def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1024 * 1024 * 5, backupCount=5
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


setup_logging()

if __name__ == "__main__":
    # slam = Slam2D(
    #     "/home/atticuszz/DevSpace/python/AutoDrive_frontend/src/camera/camera_config.yaml"
    # )
    input_dir = Path(__file__).parents[1] / "Datasets/Replica/room0"
    config_file = Path(__file__).parents[1] / "Datasets/Replica/cam_params.json"
    slam = Slam2D(input_dir.as_posix(), config_file.as_posix())
    slam.run()
