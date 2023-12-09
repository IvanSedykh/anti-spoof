import argparse
import collections
from typing import Any
import warnings
import logging

import numpy as np
import torch
from torch import nn
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
import wandb


from src.model.rawnet import RawNet
from src.utils import prepare_device, count_params, inf_loop
from src.utils.object_loading import get_datasets
from src.collate_fn.collate import collate_fn


warnings.filterwarnings("ignore", category=UserWarning)

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# fix random seeds for reproducibility
SEED = 0xDEADBEEF
torch.manual_seed(SEED)
# let's go fast boi
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

load_dotenv()


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print(config)

    # setup data_loader instances
    datasets = get_datasets(config)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.data.train.batch_size,
        shuffle=True,
        num_workers=config.data.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    model = RawNet(channels_list=config.model.channels_list)
    model.train()
    logger.info(model)
    print(f"# parameters: {count_params(model)}")


    # setup losses
    loss_fn = nn.CrossEntropyLoss()

    # setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.eps,
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=config.trainer_args.max_steps,
        num_warmup_steps=config.trainer_args.warmup_steps,
    )


    # setup accelerator
    accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)
    accelerator.init_trackers("nv_dla", config=OmegaConf.to_container(config))
    (
        model,
        optimizer,
        scheduler,
        train_loader,
    ) = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_loader,
    )

    for step, batch in enumerate(inf_loop(train_loader)):
        if step >= config.trainer_args.max_steps:
            break

        wav = batch["wav"].unsqueeze(1)
        label = batch["label"]

        # ======== train ========
        optimizer.zero_grad()

        logits = model(wav)
        loss = loss_fn(logits, label)


        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(), config.trainer_args.max_grad_norm
        )
        optimizer.step()
        scheduler.step()


        if step % config.trainer_args.logging_steps == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            accelerator.log(
                {
                    "loss": loss,
                    "lr": learning_rate,
                    "grad_norm": grad_norm,
                },
                step=step,
            )

            logger.info(
                f"step: {step}, loss: {loss},"
            )

        if step % config.trainer_args.save_steps == 0 and step > 0:
            chkpt_dir = f"{config.trainer_args.output_dir}/checkpoints/step-{step}"
            accelerator.save_model(
                model, save_directory=chkpt_dir, safe_serialization=True
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
