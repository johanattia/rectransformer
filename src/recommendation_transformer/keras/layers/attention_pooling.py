from typing import Callable, Dict, TypedDict, Union

import keras
from keras import layers
from keras import ops
from keras.backend import KerasTensor


# TODO: check Keras dtypes


class AttentionOutput(TypedDict):
    output: KerasTensor
    attention_scores: KerasTensor


class AttentionPooling(layers.Layer):
    """Attention Pooling for Predictor."""

    def __init__(
        self,
        hidden_dim: int = 64,
        activation: Union[str, Callable] = "gelu",
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # ARCHITECTURE ATTRIBUTES
        self.hidden_dim = hidden_dim
        activation = keras.activations.get(activation)
        self.activation = activation
        self.use_bias = use_bias

        # ARCHITECTURE LAYERS
        self.dense1 = layers.Dense(
            units=self.hidden_dim, activation=self.activation, use_bias=self.use_bias
        )
        self.dense2 = layers.Dense(
            units=self.hidden_dim, activation=self.activation, use_bias=self.use_bias
        )
        self.softmax = layers.Softmax(axis=1)

        # WEIGHTS
        self.conversion_query = self.add_weight(
            name="conversion_query",
            shape=[self.hidden_dim, 1],
            dtype="float32",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
            }
        )
        return config

    def call(
        self, inputs: KerasTensor, attention_mask: KerasTensor = None
    ) -> Dict[str, KerasTensor]:
        """Attention mechanism with learnable context vector.

        Args:
            inputs (tf.Tensor): client activations/touch tensors.
            attention_mask (tf.Tensor, optional): mask tensor in case of padding for example.
                Defaults to None.

        Returns:
            Dict[str, tf.Tensor]: dictionary of tensors `output` and `attention_scores`.
        """
        inputs = self.dense1(inputs)
        hidden = self.dense2(inputs)
        attention_logits = ops.einsum("BNH,HC->BNC", hidden, self.conversion_query)

        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, dtype="bool")

        attention_scores = self.softmax(attention_logits, attention_mask)
        output = ops.sum(attention_scores * inputs, axis=1)

        return AttentionOutput(output=output, attention_scores=attention_scores)
