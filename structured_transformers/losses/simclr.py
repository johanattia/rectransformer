"""Self-supervised algorithm: SimCLR with TensorFlow"""

from typing import Any, Dict

import tensorflow as tf


LARGE_NUM = 1e9


class SimCLRLoss(tf.keras.losses.Loss):
    """SimCLR loss implementation for self-supervised learning.

    Official references from Google:
    * Article: `Big Self-Supervised Models are Strong Semi-Supervised Learners` (https://arxiv.org/abs/2006.10029)
    * Code: https://github.com/google-research/simclr/blob/master/tf2/objective.py

    Example:
    ```python
    >>> import tensorflow as tf
    >>> import structured_transformers

    >>> loss_fn = structured_transformers.losses.SimCLR()
    ```
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.001,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="SimCLR",
    ):
        """_summary_

        Args:
            temperature (float, optional): _description_. Defaults to 0.05.
            margin (float, optional): _description_. Defaults to 0.001.
            reduction (_type_, optional): _description_. Defaults to tf.keras.losses.Reduction.AUTO.
            name (_type_, optional): _description_. Defaults to `SimCLR`.
        """
        super().__init__(reduction=reduction, name=name)

        self.temperature = temperature
        self.margin = margin

    def call(self, hidden1: tf.Tensor, hidden2: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            hidden1 (tf.Tensor): _description_
            hidden2 (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        batch_size = tf.shape(hidden1)[0]
        diag = tf.eye(batch_size)

        hidden1 = tf.math.l2_normalize(hidden1, axis=1)
        hidden2 = tf.math.l2_normalize(hidden2, axis=1)

        return NotImplemented

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        config = {
            "temperature": self.temperature,
            "margin": self.margin,
        }
        return dict(list(base_config.items()) + list(config.items()))
