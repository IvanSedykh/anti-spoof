import argparse
import collections
from typing import Any
import warnings
import logging

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from transformers import TrainingArguments, EvalPrediction

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import TTSTrainer, WandbPredictionProgressCallback
from src.utils import prepare_device, count_params
from src.utils.object_loading import get_metrics, get_datasets
# kinda cringe but ok for now
from src.collate_fn.collate import collate_fn as collate_fn_fastspeech1
from src.collate_fn import collate_fn_fastspeech2

warnings.filterwarnings("ignore", category=UserWarning)

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# fix random seeds for reproducibility
SEED = 0xDEADBEEF
torch.manual_seed(SEED)
# let's go fast boi
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

load_dotenv()


# todo: move to another module
class MetricsCaller:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, hf_eval_pred: EvalPrediction) -> dict[str, float]:
        preds: dict = hf_eval_pred.predictions
        preds = preds["predict_wav"]
        preds = torch.tensor(preds)
        target = hf_eval_pred.label_ids
        target = torch.tensor(target)
        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] = metric(preds, target)
        return metrics

@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config))
    print(config)

    # setup data_loader instances
    datasets = get_datasets(config)

    # build model architecture, then print to console
    model = instantiate(config.model)
    logger.info(model)
    print(f"Number of parameters: {count_params(model)}")

    # get function handles of loss and metrics
    loss_module = instantiate(config.loss)

    # metrics = get_metrics(config)
    # metrics_computer = MetricsCaller(metrics)

    trainer_args = TrainingArguments(**config.trainer_args)

    # it may be better to chose collate fn based on dataset/model, but i dont want to to extra work

    trainer = TTSTrainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets["train"],
        data_collator=collate_fn_fastspeech2
    )

    # callback = WandbPredictionProgressCallback(trainer, datasets["val"], 10)
    # trainer.add_callback(callback)

    trainer.set_loss(loss_module)

    trainer.train()


if __name__ == "__main__":
    main()
