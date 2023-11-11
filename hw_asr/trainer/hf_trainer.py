from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainerCallback
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import nested_detach
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
import wandb


class SourceSeparationTrainer(Trainer):
    def set_loss(self, loss_module: nn.Module):
        self.loss_module = loss_module

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if hasattr(self, "loss_module"):
            loss = self.loss_module(inputs=inputs, outputs=outputs)
        else:
            raise ValueError("Trainer: loss module is not set.")

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # i want to use all my fucking columns
        # data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None,
    ) -> Tuple[Tensor | None, Tensor | None, Tensor | None]:
        # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    logits = {k: v for k, v in outputs.items() if k not in ignore_keys + ["loss"]}
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    logits = {k: v for k, v in outputs.items() if k not in ignore_keys + ["loss"]}
                else:
                    logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)

        return (loss, logits, labels)




# todo: move to another module

class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer: SourceSeparationTrainer, val_dataset, num_samples: int):
        print("Init WandbPredictionProgressCallback")
        super().__init__()
        self.trainer = trainer
        self.num_samples = num_samples
        self.sample_dataset = Subset(val_dataset, indices=range(num_samples))

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        hf_output = self.trainer.predict(self.sample_dataset)
        predictions: dict = hf_output.predictions
        predictions = predictions['predict_wav']
        labels = hf_output.label_ids

        audio_predictions = [
            wandb.Audio(pred, sample_rate=16_000)
            for i, pred in enumerate(predictions)
        ]

        audio_predictions_normalized = [
            wandb.Audio(self.normalize_audio(pred), sample_rate=16_000)
            for i, pred in enumerate(predictions)
        ]
        audio_labels = [
            wandb.Audio(
                label,
                sample_rate=16_000,
            )
            for i, label in enumerate(labels)
        ]
        audio_mix = [
            wandb.Audio(
                np.array(self.sample_dataset[i]['mix_wav'][0]), sample_rate=16000,
            )
            for i in range(self.num_samples)
        ]

        predictions_df = pd.DataFrame({
             "mix": audio_mix,
             "predictions": audio_predictions, 
             "predictions_norm":audio_predictions_normalized, 
             "targets": audio_labels
        })
        predictions_df["epoch"] = state.epoch
        records_table = self._wandb.Table(dataframe=predictions_df)
        #   log the table to wandb
        self._wandb.log({"sample_predictions": records_table})

    @staticmethod
    def normalize_audio(wav: np.array):
        return 20 * wav/np.linalg.norm(wav)
