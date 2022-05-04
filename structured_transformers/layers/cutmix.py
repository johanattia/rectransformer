"""CutMix and Mixup layers with TensorFlow"""

from typing import Dict, Tuple, Union
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

    def call(
        self, inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """[summary]

        Args:
            inputs (Union[tf.Tensor, Dict[str, tf.Tensor]]): [description]

        Returns:
            Union[tf.Tensor, Dict[str, tf.Tensor]]: [description]
        """
        binomial_seeds = tf.random.uniform(
            shape=(2,), minval=0, maxval=self.seed, dtype=tf.int32, seed=self.seed
        )

        if isinstance(inputs, dict):
            shape, batch_size = self.nested_shape(inputs)
            binomial_masks = tf.random.stateless_binomial(
                shape, binomial_seeds, counts=1, probs=self.probability
            )
            shuffled_indices = tf.random.shuffle(tf.range(batch_size), seed=self.seed)

            output = {}
            for key in inputs.keys():
                shuffled_inputs = tf.gather(inputs[key], indices=shuffled_indices)
                output[key] = inputs[key] * binomial_masks + shuffled_inputs * (
                    1 - binomial_masks
                )

        elif isinstance(inputs, tf.Tensor):
            binomial_masks = tf.random.stateless_binomial(
                tf.shape(inputs), binomial_seeds, counts=1, probs=self.probability
            )
            shuffled_inputs = tf.random.shuffle(inputs, seed=self.seed)
            output = inputs * binomial_masks + shuffled_inputs * (1 - binomial_masks)

        else:
            raise ValueError(
                "`inputs` batch must be either a tf.Tensor or a nested dictionary batch of tf.Tensor."
            )

        return output

    def nested_shape(self, inputs: Dict[str, tf.Tensor]) -> Tuple[tf.TensorShape, int]:
        """[summary]

        Args:
            inputs (Dict[str, tf.Tensor]): [description]

        Returns:
            Tuple[tf.TensorShape, int]: [description]
        """
        key = next(iter(inputs))
        shape = tf.shape(inputs[key])
        batch_size = shape[0]

        return shape, batch_size

    def get_config(self) -> dict:
        base_config = super(CutMix, self).get_config()
        config = {
            "probability": self.probability,
            "seed": self.seed,
        }
        return dict(list(base_config.items()) + list(config.items()))
