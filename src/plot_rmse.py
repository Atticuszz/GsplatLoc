from src.eval.logger import WandbLogger

if __name__ == "__main__":
    log = WandbLogger(run_name="plot_RMSE", config={"description": "plots"})

    log.plot_RMSE()
