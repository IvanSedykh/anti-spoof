import argparse
import collections
from typing import Any
import warnings

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import TrainingArguments, EvalPrediction

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import SourceSeparationTrainer, WandbPredictionProgressCallback
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_metrics, get_datasets
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.collate_fn.collate import collate_fn

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 0xDEADBEEF
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    datasets = get_datasets(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss)

    metrics = get_metrics(config)
    metrics_computer = MetricsCaller(metrics)

    config["trainer_args"]["output_dir"] = config._save_dir
    trainer_args = TrainingArguments(**config["trainer_args"])

    trainer = SourceSeparationTrainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=collate_fn,
        compute_metrics=metrics_computer,
    )

    callback = WandbPredictionProgressCallback(trainer, datasets["val"], 10)
    trainer.add_callback(callback)

    trainer.set_loss(loss_module)

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
