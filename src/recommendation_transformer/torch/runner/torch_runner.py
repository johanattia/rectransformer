"""Lightweight Torch Module Runner for Training, Evaluation and Inference"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch import optim

import keras
from keras import backend
from keras import metrics as metrics_module

from recommendation_transformer.torch import callbacks


# TODO: train + _append
# device:
# https://stackoverflow.com/questions/75188254/pytorch-proper-way-to-compute-loss-on-gpu
# https://docs.databricks.com/en/machine-learning/model-inference/resnet-model-inference-pytorch.html
# Review loss weighting in update_metrics, test_step...


def _copy_loss(loss_obj, reduction: str):
    obj_copy = loss_obj.__class__()
    obj_copy.__dict__.update(loss_obj.__dict__)
    obj_copy.reduction = reduction
    return obj_copy


def _append(batch_output, output):
    return output


@dataclass
class RunCounter:
    train: int = 0
    evaluate: int = 0
    predict: int = 0


class TorchRunner:
    """Lightweight Torch Module Runner for Training, Evaluation and Inference"""

    def __init__(
        self,
        model: nn.Module,
        jit_compile: bool,
        device: torch.device,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        metrics: Iterable[metrics_module.Metric],
    ):
        _backend = backend.backend()
        if _backend != "torch":
            raise ValueError(f"Keras backend should be `torch`. Got `{_backend}`")

        if jit_compile and not isinstance(model, torch._dynamo.OptimizedModule):
            self.model = torch.compile(model)
        else:
            self.model = model

        self._jit_compile = jit_compile
        self._device = device
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss_tracker = metrics_module.Mean(name="loss")

        self.sample_loss_fn = (
            loss_fn
            if loss_fn.reduction == "none"
            else _copy_loss(loss_fn, reduction="none")
        )
        self.run_counter = RunCounter()

    @property
    def device(self):
        return self._device

    @property.setter
    def device(self, value: torch.device):
        self._device = value

    @property
    def loss_fn(self):
        return self._loss_fn

    @property.setter
    def loss_fn(self, value: nn.Module):
        self._loss_fn = value

    @property
    def optimizer(self):
        return self._optimizer

    @property.setter
    def optimizer(self, value: optim.Optimizer):
        if not isinstance(value, optim.Optimizer):
            raise TypeError
        self._optimizer = value

    def reset_counter(self):
        self.run_counter = RunCounter()

    def reset_metrics(self, include_loss: bool = True):
        for metric in self.metrics():
            metric.reset_state()
        if include_loss:
            self._loss_tracker.reset_state()

    def update_metrics(
        self,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        sample_weight: torch.Tensor = None,
        losses: torch.Tensor = None,
        loss_weighting: bool = True,
    ):
        for metric in self.metrics:
            metric.update_state(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )
        if losses is not None:
            loss_weight = sample_weight if loss_weighting else None
            self._loss_tracker.update_state(values=losses, sample_weight=loss_weight)

    def compute_metrics(self, include_loss: bool = True) -> List[Tuple[str, float]]:
        results = [("loss", self._loss_tracker.result())] if include_loss else []
        results += [(f"{metric.name}", metric.result()) for metric in self.metrics]
        return results

    def compute_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: torch.Tensor = None,
        auto_reduction: bool = True,
    ) -> torch.Tensor:
        if not auto_reduction:
            return self.sample_loss_fn(y_pred, y_true)

        if sample_weight is None:
            return self.loss_fn(y_pred, y_true)

        weighted_losses = self.sample_loss_fn(y_pred, y_true) * sample_weight

        if self.loss_fn.reduction == "none":
            return weighted_losses
        else:
            reduce_fn = torch.sum if self.loss_fn.reduction == "sum" else torch.mean
            return reduce_fn(weighted_losses)

    def test_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        self.model.eval()

        # Inference and loss
        with torch.no_grad():
            outputs = self.model(inputs)
            losses = self.compute_loss(
                y_true=targets, y_pred=outputs, auto_reduction=False
            )
        # Weighted loss and metrics computation
        self.update_metrics(
            targets=targets, outputs=outputs, losses=losses, sample_weight=sample_weight
        )
        results = self.compute_metrics(include_loss=True)

        return results, outputs

    def predict_step(self, inputs: torch.Tensor):
        self.model.eval()

        # Inference
        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs

    def train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor
    ):
        self.model.train()
        self.optimizer.zero_grad()

        # Inference and loss
        outputs = self.model(inputs)
        loss = self.compute_loss(outputs, targets, sample_weight)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        # Weighted loss and metrics computation
        self.update_metrics(
            targets=targets, outputs=outputs, losses=loss, sample_weight=sample_weight
        )
        results = self.compute_metrics(include_loss=True)

        return results

    def evaluate(self, test_dataloader: torch.utils.data.DataLoader, verbose: int = 1):
        self.reset_metrics(include_loss=True)

        iterations = len(test_dataloader)
        progbar = keras.utils.Progbar(
            targets=iterations,
            width=30,
            verbose=verbose,
            stateful_metrics=["loss"] + [metric.name for metric in self.metrics],
        )

        outputs = None
        for step, batch in enumerate(test_dataloader):
            # Load data
            if len(batch) == 2:
                (inputs, targets), sample_weight = batch, None
            else:
                inputs, targets, sample_weight = batch

            # Inference and metrics computation are performed
            results, batch_output = self.test_step(inputs, targets, sample_weight)
            # Structure final outputs composed of batches of outputs
            outputs = _append(batch_output, outputs)
            progbar.update(step + 1, values=results, finalize=False)

        progbar.update(step + 1, values=results, finalize=True)
        self.run_counter.evaluate += 1

        return {"results": results, "outputs": outputs}

    def predict(self, dataloader: torch.utils.data.DataLoader, verbose: int = 1):
        iterations = len(dataloader)
        progbar = keras.utils.Progbar(targets=iterations, width=30, verbose=verbose)

        outputs = None
        for step, batch_input in enumerate(dataloader):
            # Inference
            batch_output = self.predict_step(batch_input)
            # Structure final outputs composed of batches of outputs
            outputs = _append(batch_output, outputs)
            progbar.update(step + 1, finalize=False)

        progbar.update(step + 1, finalize=True)
        self.run_counter.predict += 1

        return outputs

    def train(
        self,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader = None,
        callbacks: Iterable[callbacks.Callback] = None,
        verbose: int = 1,
    ):
        history = {}
        ## TRAINING PROCEDURE
        # FOR EACH EPOCH:
        ### FOR EACH TRAIN ITERATION: TRAIN_STEP+SAVE TRAIN METRICS
        ### FOR EACH VAL ITERATION: TEST_STEP+SAVE VAL METRICS WITH VERBOSE=2
        ### HISTORY UPDATE TRAIN&VAL LOSSES AND METRICS
        self.run_counter.train += 1

        return history
