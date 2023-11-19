from operator import xor

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from omegaconf import DictConfig
from hydra.utils import instantiate


import src.augmentations
import src.datasets
from src.collate_fn.collate import make_collate_fn_tensor

import src.metric as module_metric


def get_datasets(config: DictConfig) -> dict[str, Dataset]:
    datasets_dict = {}
    for split, params in config["data"].items():

        # todo: imlement augmentations

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(
                instantiate(ds)
            )
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        
        if hasattr(params, 'limit'):
            dataset = Subset(dataset, range(params.limit))
        datasets_dict[split] = dataset
    return datasets_dict


def get_metrics(config: DictConfig) -> dict[str, list]:
    common_metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"].get("all", [])
    ]
    # ignore other metrics
    return common_metrics


# def get_metrics(config, text_encoder):
#     evalutation_metrics = [
#         config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
#         for metric_dict in config["metrics"].get("evaluation_metrics", [])
#     ]
#     train_metrics = [
#         config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
#         for metric_dict in config["metrics"].get("train_metrics", [])
#     ]

#     metrics = {
#         "train_metrics": common_metrics + train_metrics,
#         "evaluation_metrics": common_metrics + evalutation_metrics,
#     }

#     return metrics
