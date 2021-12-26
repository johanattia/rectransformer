"""CutMix and Mixup layers with TensorFlow"""

import tensorflow as tf


class CutMix(tf.keras.layers.Layer):
    """[summary]

    Args:
        probability (float): [description]
        seed (int, optional): [description]. Defaults to None.
    """

    def __init__(self, probability: float, seed: int = None, **kwargs):
        super(CutMix, self).__init__(**kwargs)

        self.probability = probability
        self.seed = seed

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Not yet implemented")

    def get_config(self) -> dict:
        base_config = super(CutMix, self).get_config()
        config = {
            "probability": self.probability,
            "seed": self.seed,
        }
        return dict(list(base_config.items()) + list(config.items()))


class Mixup(tf.keras.layers.Layer):
    """[summary]

    Args:
        alpha (float): [description]
    """

    def __init__(self, alpha: float, **kwargs):
        super(Mixup, self).__init__(**kwargs)

        self.alpha = alpha

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Not yet implemented")

    def get_config(self) -> dict:
        base_config = super(Mixup, self).get_config()
        config = {
            "alpha": self.alpha,
        }
        return dict(list(base_config.items()) + list(config.items()))
