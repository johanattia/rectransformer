"""Orbit-like Torch Module Runner/Controller: https://www.tensorflow.org/api_docs/python/orbit"""

from typing import Callable, Dict, Iterable, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from recommendation_transformer.torch.callbacks import callback


class TorchRunner:
    """Lightweight single-runner object for training, evaluation and inference."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Iterable[Callable],
        jit_compile: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.jit_compile = jit_compile

        self._loss_tracker = None
        self._metrics_tracker = None

        self._train_dataloader = None
        self._validation_dataloader = None
        self._callbacks = None

        self._completed_train_counter = 0

    def reset_metrics(self):
        pass

    def _update_metrics(self):
        pass

    def compute_metrics(self):
        pass

    def reset_loss(self):
        pass

    def _update_loss(self):
        pass

    def compute_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        if sample_weight is None:
            return self.criterion(y_true, y_pred)

        if self.criterion.reduction == "none":
            return self.criterion(y_true, y_pred) * sample_weight
        else:
            criterion = self.criterion.__class__()
            criterion.__dict__.update(self.criterion.__dict__)
            criterion.reduction = "none"

            sample_loss = criterion(y_true, y_pred) * sample_weight
            reduce_op = torch.sum if self._criterion.reduction == "sum" else torch.mean

            return reduce_op(sample_loss)

    def loss_and_metrics_results(self):
        return {"loss": self._loss_tracker, **self._metrics_tracker}

    def test_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.compute_loss(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )
            metrics = self.compute_metrics(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )

        return loss, metrics

    def predict_step(self):
        pass

    def train_step(
        self,
    ):
        pass

    def evaluate(self, validation_dataloader: DataLoader):
        self.model.eval()
        self.reset_loss()
        self.reset_metrics()

        for batch in validation_dataloader:
            if len(batch) == 2:
                (inputs, targets), sample_weight = batch, None
            else:
                inputs, targets, sample_weight = batch

            validation_loss, validation_metrics = self.test_step(
                inputs, targets, sample_weight
            )

            self._update_loss(validation_loss)
            self._update_metrics(validation_metrics)

        return self.loss_and_metrics_results()

    def predict(self):
        self.model.eval()
        return

    def train(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader = None,
        callbacks: Iterable[callback.Callback] = None,
    ):
        self.model.train()

        self.completed_train_counter += 1
        return
