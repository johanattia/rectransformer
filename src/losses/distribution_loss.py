"""Modified entropy-related loss for distributional learning"""


from typing import Callable, Union
import tensorflow as tf


class EntropyDistributionLoss(tf.keras.losses.Loss):
    """_summary_

    Args:
        entropy_distance (Union[str, Callable], optional): _description_.
            Defaults to None.
        from_logits (bool, optional): _description_.
            Defaults to False.
        reduction (str, optional): _description_.
            Defaults to tf.keras.losses.Reduction.AUTO.
        name (str, optional): _description_.
            Defaults to None.

    Raises:
        ValueError: _description_
    """

    def __init__(
        self,
        entropy_distance: Union[str, Callable] = None,
        from_logits: bool = False,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = None,
    ):
        super().__init__(reduction, name)

        if entropy_distance is None:
            entropy_distance = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            )
        elif isinstance(entropy_distance, str):
            entropy_distance = tf.keras.losses.get(entropy_distance)
        elif isinstance(entropy_distance, tf.keras.losses.Loss):
            entropy_distance.reduction = tf.keras.losses.Reduction.NONE
        elif not callable(entropy_distance):
            raise ValueError(
                "Please give a valid value for `entropy_distance` argument."
            )
        self._entropy_distance = entropy_distance
        self._from_logits = from_logits

    @staticmethod
    def _compute_entropy(distribution: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.categorical_crossentropy(
            y_true=distribution, y_pred=distribution
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
        return distribution_divergence + entropy_distance
