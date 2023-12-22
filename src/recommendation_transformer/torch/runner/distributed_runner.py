from typing import Iterable

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from recommendation_transformer.torch.callbacks import callback
from recommendation_transformer.torch.runner import TorchRunner


class TorchDistributedRunner(TorchRunner):
    pass
