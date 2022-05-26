"""Modified entropy-related loss for distributional learning"""


from typing import Callable, Optional, Union
import tensorflow as tf


class EntropicDistributionLoss(tf.keras.losses.Loss):
    """Initializes a distributional loss instance based on
    Kullback-Leibler divergence and entropies distance.

    Args:
        entropy_distance (tf.keras.losses.Loss): Error function used to compute
            distance between `y_true` and `y_pred` distributions entropies.
            Defaults to mean squared error.
        from_logits (bool): Whether `y_pred` is expected to be a logits tensor.
            By default, we assume that `y_pred` encodes a probability distribution.
        reduction (str):  Type of `tf.keras.losses.Reduction` to apply to loss.
            Defaults to tf.keras.losses.Reduction.AUTO.
        name (str): Optional name for the instance.
            Defaults to None.
    """

    def __init__(
        self,
        entropy_distance: Optional[tf.keras.losses.Loss] = None,
        from_logits: bool = False,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = None,
    ):
        super().__init__(reduction, name)

        entropy_distance = tf.keras.losses.get(entropy_distance)
        self._entropy_distance = (
            entropy_distance
            if entropy_distance is not None
            else tf.keras.losses.MeanSquaredError()
        )
        self._entropy_distance.reduction = tf.keras.losses.Reduction.NONE

        self._from_logits = from_logits

    @staticmethod
    def _compute_entropy(dist: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.categorical_crossentropy(
            y_true=dist, y_pred=dist, from_logits=False
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        if self._from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true_entropy = self._compute_entropy(y_true)
        y_pred_entropy = self._compute_entropy(y_pred)

        entropy_distance = self._entropy_distance(
            y_true=y_true_entropy, y_pred=y_pred_entropy
        )
        distribution_divergence = tf.keras.losses.kl_divergence(
            y_true=y_true, y_pred=y_pred
        )
        return entropy_distance + distribution_divergence

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "entropy_distance": tf.keras.losses.serialize(self._entropy_distance),
                "from_logits": self._from_logits,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["entropy_distance"] = tf.keras.losses.get(config["entropy_distance"])
        return cls(**config)
