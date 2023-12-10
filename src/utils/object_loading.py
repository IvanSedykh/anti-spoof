from operator import xor

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.utils import to_absolute_path




def get_datasets(config: DictConfig) -> dict[str, Dataset]:
    datasets_dict = {}
    for split, params in config["data"].items():
        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            # change root path to absolute path
            ds["data_path"] = to_absolute_path(ds["data_path"])
            datasets.append(instantiate(ds))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        if hasattr(params, "limit"):
            dataset = Subset(dataset, range(params.limit))
        print(f"dataset {split} length = {len(dataset)}")
        datasets_dict[split] = dataset
    return datasets_dict
