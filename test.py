import argparse
import json
import os
from pathlib import Path
import datetime

import torch
from tqdm import tqdm
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import TrainingArguments, EvalPrediction


import src.model as module_model
from src.trainer import TTSTrainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders, get_datasets, get_metrics
from src.utils.parse_config import ConfigParser
from src.collate_fn.collate import collate_fn

# todo:
from train import MetricsCaller


# fix random seeds for reproducibility
SEED = 0xDEADBEEF
torch.manual_seed(SEED)
np.random.seed(SEED)

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def load_safetensors_dict(path: Path, device):
    tensors = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # init metrics
    metrics = get_metrics(config)
    metrics_computer = MetricsCaller(metrics)

    # setup data_loader instances
    print(f"Loading data...")
    datasets = get_datasets(config)
    print(datasets["test"])

    # build model architecture
    print(f"Building model...")
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    checkpoint_path = config.resume + "/model.safetensors"
    logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
    # state_dict = load_safetensors_dict(checkpoint_path, device)
    state_dict = load_file(checkpoint_path, "cpu")
    model.load_state_dict(state_dict)

    trainer_args = TrainingArguments(**config["trainer_args"])
    trainer = TTSTrainer(
        model=model,
        args=trainer_args,
        data_collator=collate_fn,
        compute_metrics=metrics_computer,
    )

    res = trainer.predict(datasets["test"])

    print(res.metrics)


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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirTestDataset",
                        "args": {"data_dir": str(test_data_folder)},
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
