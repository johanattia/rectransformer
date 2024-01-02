"""Lightweight Torch Module Runner for Training, Evaluation and Inference"""

import os
from typing import Iterable

os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch import nn
import keras

from recommendation_transformer.torch import callbacks


# TODO;
# handling sample_weight argument
# finalize compute_loss method with per_sample arg


class TorchRunner:
    """Lightweight Torch Module Runner for Training, Evaluation and Inference"""

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

    def reset_metrics(self, include_loss: bool = True):
        for metric in self.metrics():
            metric.reset_state()

        if include_loss:
            self._loss_tracker.reset_state()

    def update_metrics(
        self,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        losses: torch.Tensor = None,
        sample_weight: torch.Tensor = None,
    ):
        for metric in self.metrics:
            metric.update_state(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )
        if losses is not None:
            self._loss_tracker.update_state(values=losses, sample_weight=sample_weight)

    def compute_metrics(self, include_loss: bool = True):
        results = [("loss", self._loss_tracker.result())] if include_loss else []
        results += [(f"{metric.name}", metric.result()) for metric in self.metrics]
        return results

    def compute_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: torch.Tensor = None,
        per_sample: bool = False,
    ) -> torch.Tensor:
        if per_sample:
            loss_fn = self.loss_fn.__class__()
            loss_fn.__dict__.update(self.loss_fn.__dict__)
            loss_fn.reduction = "none"

            return loss_fn(y_pred, y_true)

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

    def test_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(inputs)
            losses = self.compute_loss(y_true=targets, y_pred=outputs, per_sample=True)

        return outputs, losses

    def predict_step(self):
        self.model.eval()
        pass

    def train_step(
        self,
    ):
        self.model.train()
        pass

    def evaluate(self, test_dataloader: torch.utils.data.DataLoader):
        self.reset_metrics(include_loss=True)

        iterations = len(test_dataloader)
        progbar = keras.utils.Progbar(
            targets=iterations,
            width=30,
            stateful_metrics=["loss"] + [metric.name for metric in self.metrics],
        )

        for step, batch in enumerate(test_dataloader):
            if len(batch) == 2:
                (inputs, targets), sample_weight = batch, None
            else:
                inputs, targets, sample_weight = batch

            outputs, losses = self.test_step(inputs, targets)

            self.update_metrics(
                targets=targets,
                outputs=outputs,
                sample_weight=sample_weight,
                losses=losses,
            )
            results = self.compute_metrics(include_loss=True)
            progbar.update(step + 1, values=results, finalize=True)

        progbar.update(step + 1, values=results, finalize=True)
        return results

    def predict(self):
        return

    def train(
        self,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader = None,
        callbacks: Iterable[callbacks.Callback] = None,
    ):
        history = {}
        ## TRAINING PROCEDURE
        # FOR EACH EPOCH:
        ### FOR EACH TRAIN ITERATION: TRAIN_STEP+SAVE TRAIN METRICS
        ### FOR EACH VAL ITERATION: TEST_STEP+SAVE VAL METRICS
        ### HISTORY UPDATE TRAIN&VAL LOSSES AND METRICS
        self._train_counter += 1
        return history
