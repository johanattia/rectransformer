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
        """[summary]

        Args:
            inputs (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        binomial_seeds = tf.random.uniform(
            shape=(2,), minval=0, maxval=self.seed, dtype=tf.int32, seed=self.seed
        )
        binomial_masks = tf.random.stateless_binomial(
            inputs.shape, binomial_seeds, counts=1, probs=self.probability
        )

        shuffled_inputs = tf.random.shuffle(inputs, seed=self.seed)
        output = inputs * binomial_masks + shuffled_inputs * (1 - binomial_masks)

        return output

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
