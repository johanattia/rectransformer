"""Orbit-like Torch Module Runner/Controller"""

from typing import Iterable

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from recommendation_transformer.torch.callbacks import callback


class TorchRunner:
    """Lightweight single-runner object for training, evaluation and inference.

    Orbit:
    - https://www.tensorflow.org/tfmodels/orbit
    - https://www.tensorflow.org/api_docs/python/orbit
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self._optimizer = None
        self._callbacks = None

    def predict_step(self):
        pass

    def predict(self):
        pass

    def test_step(self):
        pass

    def evaluate(self):
        pass

    def train_step(self):
        pass

    def train(
        self,
        epochs: int,
        optimizer: optim.Optimizer,
        callbacks: Iterable[callback.Callback] = None,
        # dataset, loss, metrics, compile...
    ):
        pass
