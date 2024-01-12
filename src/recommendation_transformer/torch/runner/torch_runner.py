"""Lightweight Torch Module Runner for Training, Evaluation and Inference"""

from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch import optim

import keras
from keras import backend

from recommendation_transformer.torch import callbacks


# TODO: train + _append
# device:
# https://stackoverflow.com/questions/75188254/pytorch-proper-way-to-compute-loss-on-gpu
# https://docs.databricks.com/en/machine-learning/model-inference/resnet-model-inference-pytorch.html


def _append(batch_output, output):
    return output


class TorchRunner:
    """Lightweight Torch Module Runner for Training, Evaluation and Inference"""

    def __init__(
        self,
        model: nn.Module,
        jit_compile: bool,
        device: torch.device,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        metrics: Iterable[keras.metrics.Metric],
        loss_tracker: keras.metrics.Metric = keras.metrics.Mean(),
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
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if loss_fn.reduction == "none":
            sample_loss_fn = loss_fn
        else:
            sample_loss_fn = self.loss_fn.__class__()
            sample_loss_fn.__dict__.update(self.loss_fn.__dict__)
            sample_loss_fn.reduction = "none"

        self._sample_loss_fn = sample_loss_fn

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
        loss: torch.Tensor = None,
        sample_weight: torch.Tensor = None,
    ):
        for metric in self.metrics:
            metric.update_state(
                y_true=targets, y_pred=outputs, sample_weight=sample_weight
            )
        if loss is not None:
            self._loss_tracker.update_state(values=loss, sample_weight=sample_weight)

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

        # Metrics
        self.update_metrics(
            targets=targets, outputs=outputs, loss=losses, sample_weight=sample_weight
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

        # Metrics
        self.update_metrics(
            targets=targets, outputs=outputs, loss=loss, sample_weight=sample_weight
        )
        results = self.compute_metrics(include_loss=True)

        return results

    def evaluate(self, test_dataloader: torch.utils.data.DataLoader):
        self.reset_metrics(include_loss=True)

        iterations = len(test_dataloader)
        progbar = keras.utils.Progbar(
            targets=iterations,
            width=30,
            stateful_metrics=["loss"] + [metric.name for metric in self.metrics],
        )

        outputs = None
        for step, batch in enumerate(test_dataloader):
            if len(batch) == 2:
                (inputs, targets), sample_weight = batch, None
            else:
                inputs, targets, sample_weight = batch

            results, batch_output = self.test_step(inputs, targets, sample_weight)
            outputs = _append(batch_output, outputs)
            progbar.update(step + 1, values=results, finalize=False)

        progbar.update(step + 1, values=results, finalize=True)

        return {"results": results, "outputs": outputs}

    def predict(self, dataloader: torch.utils.data.DataLoader):
        iterations = len(dataloader)
        progbar = keras.utils.Progbar(targets=iterations, width=30)

        outputs = None
        for step, batch_input in enumerate(dataloader):
            batch_output = self.predict_step(batch_input)
            outputs = _append(batch_output, outputs)
            progbar.update(step + 1, finalize=False)

        progbar.update(step + 1, finalize=True)

        return outputs

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
