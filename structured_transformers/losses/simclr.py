"""Self-supervised algorithm: SimCLR with TensorFlow"""

from typing import Any, Dict

import tensorflow as tf


LARGE_NUM = 1e9


class SimCLR(tf.keras.losses.Loss):
    """SimCLR loss for self-supervised learning. Official reference from Google:
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

    def call(self, feature1: tf.Tensor, feature2: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            feature1 (tf.Tensor): _description_
            feature2 (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """

        batch_size = tf.shape(feature1)[0]
        diag = tf.eye(batch_size)

        return NotImplemented

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        config = {
            "temperature": self.temperature,
            "margin": self.margin,
        }
        return dict(list(base_config.items()) + list(config.items()))
