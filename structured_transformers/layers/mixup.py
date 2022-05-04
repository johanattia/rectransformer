"""CutMix and Mixup layers with TensorFlow"""

import tensorflow as tf


class Mixup(tf.keras.layers.Layer):
    """[summary]

    Args:
        alpha (float): [description]
        seed (int, optional): [description]. Defaults to None.
    """

    def __init__(self, alpha: float, seed: int = None, **kwargs):
        super(Mixup, self).__init__(**kwargs)

        self.alpha = alpha
        self.seed = seed

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        shuffled_inputs = tf.random.shuffle(inputs, seed=self.seed)
        output = self.alpha * inputs + (1 - self.alpha) * shuffled_inputs

        return output

    def get_config(self) -> dict:
        base_config = super(Mixup, self).get_config()
        config = {
            "alpha": self.alpha,
            "seed": self.seed,
        }
        return dict(list(base_config.items()) + list(config.items()))
