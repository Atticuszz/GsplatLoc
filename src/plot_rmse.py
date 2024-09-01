import sys
from pathlib import Path

ROOT = Path(__file__).parents[1].as_posix()
sys.path.append(ROOT)
from src.eval.logger import WandbLogger

if __name__ == "__main__":
    baseline = "replica_icps"
    # baseline = "tum"
    log = WandbLogger(
        run_name="plot_RMSE", config={"description": f"baseline_{baseline}"}
    )

    log.plot_RMSE(f"baseline_{baseline}")
