from pathlib import Path
import sys
ROOT = Path(__file__).parents[1].as_posix()
sys.path.append(ROOT)
from src.eval.logger import WandbLogger

if __name__ == "__main__":
    log = WandbLogger(run_name="plot_RMSE", config={"description": "plots"})

    log.plot_RMSE('gsplatloc_2')
