"""Keras-like callbacks for PyTorch"""


import torch.nn as nn
import torch.nn.functional as F


class Callback:
    pass


class EarlyStopping(Callback):
    pass


class ModelCheckpoint(Callback):
    pass


class TensorBoard(Callback):
    pass
