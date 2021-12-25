"""CutMix and Mixup layers with TensorFlow"""

import tensorflow as tf


class CutMixLayer(tf.keras.layers.Layer):
    def __init__(self, probability: float, seed: int, **kwargs):
        """[summary]

        Args:
            probability (float): [description]
            seed (int): [description]
        """
        super(CutMixLayer, self).__init__(**kwargs)

        self.probability = probability
        self.seed = seed

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Not yet implemented")

    def get_config(self) -> dict:
        base_config = super(CutMixLayer, self).get_config()
        config = {
            "probability": self.probability,
            "seed": self.seed,
        }
        return dict(list(base_config.items()) + list(config.items()))


class MixupLayer(tf.keras.layers.Layer):
    """[summary]

    Args:
        alpha (float): [description]
    """

    def __init__(self, alpha: float, **kwargs):
        super(MixupLayer, self).__init__(**kwargs)

        self.alpha = alpha

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Not yet implemented")

    def get_config(self) -> dict:
        base_config = super(MixupLayer, self).get_config()
        config = {
            "alpha": self.alpha,
        }
        return dict(list(base_config.items()) + list(config.items()))
