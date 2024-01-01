"""Orbit-like Torch Module Runner/Controller: https://www.tensorflow.org/api_docs/python/orbit"""

import os
from typing import Dict, Iterable, Tuple, Union

os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch import nn
import keras

from recommendation_transformer.torch import callbacks


class TorchRunner:
    """Lightweight single-runner for training, evaluation and inference."""

    def __init__(
        self,
        model: nn.Module,
        jit_compile: bool,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_tracker: keras.metrics.Metric,
        metrics: Iterable[keras.metrics.Metric],
    ):
        if jit_compile and not isinstance(model, torch._dynamo.OptimizedModule):
            self.model = torch.compile(model)
        else:
            self.model = model

        self.jit_compile = jit_compile
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if isinstance(loss_tracker, (keras.metrics.Mean, keras.metrics.Sum)):
            self._loss_tracker = loss_tracker
        else:
            raise ValueError(
                "`loss_tracker` should be keras.metrics.Mean or keras.metrics.Sum."
            )
        self.metrics = metrics

        self._train_dataloader = None
        self._validation_dataloader = None
        self._callbacks = None

        self._train_counter = 0

    def reset_metrics(self):
        pass

    def _update_metrics(self):
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
            return self.loss_fn(y_pred, y_true)

        if self.loss_fn.reduction == "none":
            return self.loss_fn(y_pred, y_true) * sample_weight
        else:
            loss_fn = self.loss_fn.__class__()
            loss_fn.__dict__.update(self.loss_fn.__dict__)
            loss_fn.reduction = "none"

            sample_loss = loss_fn(y_pred, y_true) * sample_weight
            reduce_fn = torch.sum if self.loss_fn.reduction == "sum" else torch.mean

            return reduce_fn(sample_loss)

    def loss_and_metrics_results(self):
        return {
            "loss": self._loss_tracker.result(),
            **{metric.name: metric.result() for metric in self.metrics},
        }

    # sample_weight à revoir
    def test_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.compute_loss(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )
        return outputs, loss

    def predict_step(self):
        pass

    def train_step(
        self,
    ):
        pass

    # sample_weight à revoir
    def evaluate(self, validation_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        self.reset_loss()
        self.reset_metrics()

        # progbar = keras.utils.Progbar()

        for batch in validation_dataloader:
            if len(batch) == 2:
                (inputs, targets), sample_weight = batch, None
            else:
                inputs, targets, sample_weight = batch

            outputs, loss = self.test_step(inputs, targets, sample_weight)

            self.update_loss(loss, sample_weight)
            self.update_metrics(targets, outputs, sample_weight)

        return self.loss_and_metrics_results()

    def predict(self):
        self.model.eval()
        return

    def train(
        self,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader = None,
        callbacks: Iterable[callbacks.Callback] = None,
    ):
        self.model.train()

        self._train_counter += 1
        return
