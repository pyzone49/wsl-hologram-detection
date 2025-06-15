import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import mlflow
from src.utils.utils import already_run, mlruntodict, get_metrics
import logging
from lightning import seed_everything


# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf")
def main(cfg : DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    # print(OmegaConf.to_yaml(cfg))
    task_name_full = HydraConfig.get().runtime.choices.decision + "_" + cfg.task_name
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    cfg.tuner.experiment_name = task_name_full
    tuner = instantiate(cfg.tuner)

    run = tuner.getBestRun()

    params = mlruntodict(run.data.params)
    params = DictConfig(params)

    decision = instantiate(params.decision)
    total_metrics = []
    mlflow.set_experiment("test_"+cfg.paths.split_name)
    if not already_run(cfg, "test_"+cfg.paths.split_name):
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", f"{task_name_full}")
            # test model
            mlflow.log_params(params)
            for test_d in cfg.data.test:
                seed_everything(cfg.seed, workers=True)
                model = instantiate(params.model)
                decision = instantiate(params.decision)
                data_test = instantiate(cfg.data.test[test_d])
                metrics = get_metrics(data_test, model, decision)
                total_metrics.append(metrics)
                mlflow.log_metrics({f"{test_d}_{k}":metrics[k] for k in metrics})
            total_metrics_str = {f"{test_d}_{k}":metrics[k] for k in metrics}
            mlflow.log_text(f"{task_name_full} {test_d} {total_metrics_str}", "metrics.txt")
    else:
        log.info("this run was already lauched")
    print(total_metrics)

if __name__ == "__main__":
    main()