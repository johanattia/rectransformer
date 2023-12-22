from typing import Callable, TypedDict, Union

import keras
from keras import layers
from keras.backend import KerasTensor


class EncoderBlockInput(TypedDict, total=False):
    input: KerasTensor
    attention_mask: KerasTensor


class DecoderBlockInput(TypedDict, total=False):
    input: KerasTensor
    context: KerasTensor
    attention_mask: KerasTensor
    cross_attention_mask: KerasTensor


class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        inner_dim: int = None,
        output_dim: int = None,
        norm_first: bool = True,
        activation: Union[Callable, str] = "gelu",
        output_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        inner_dropout: float = 0.0,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        # ARCHITECTURE ATTRIBUTES
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim if inner_dim is not None else embed_dim
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self._norm_first = norm_first
        self.output_dropout = output_dropout
        self.attention_dropout = attention_dropout
        self.inner_dropout = inner_dropout
        self.epsilon = epsilon

        activation = keras.activations.get(activation)
        self.activation = activation

        # ARCHITECTURE LAYERS
        self.attention_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.attention_dropout,
            use_bias=True,
        )

        inner_config = {
            "units": self.inner_dim,
            "activation": self.activation,
            "use_bias": True,
        }
        output_config = {
            "units": self.output_dim,
            "activation": self.activation,
            "use_bias": True,
        }
        self.ffn_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.ffn = keras.Sequential(
            [
                layers.Dense(**inner_config),
                layers.Dropout(rate=self.inner_dropout),
                layers.Dense(**output_config),
                layers.Dropout(rate=self.output_dropout),
            ]
        )

    def get_config(self):
        activation = keras.activations.serialize(self.activation)
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "inner_dim": self.inner_dim,
                "output_dim": self.output_dim,
                "norm_first": self._norm_first,
                "activation": activation,
                "output_dropout": self.output_dropout,
                "attention_dropout": self.attention_dropout,
                "inner_dropout": self.inner_dropout,
                "epsilon": self.epsilon,
            }
        )
        return config

    def call(self, inputs: EncoderBlockInput, training: bool) -> KerasTensor:
        # Inputs
        input_tensor = inputs["input"]
        attention_mask = inputs.get("attention_mask", None)

        if self._norm_first:
            # Attention block
            input_norm = self.attention_norm(input_tensor)
            attention_output = self.attention(
                query=input_norm,
                value=input_norm,
                attention_mask=attention_mask,
                training=training,
            )
            attention_output = input_tensor + attention_output

            # Feed forward block
            output_norm = self.ffn_norm(attention_output)
            output = attention_output + self.ffn(output_norm)
        else:
            # Attention block
            attention_output = self.attention(
                query=input_tensor,
                value=input_tensor,
                attention_mask=attention_mask,
                training=training,
            )
            attention_output = self.attention_norm(input_tensor + attention_output)

            # Feed forward block
            output = self.ffn(attention_output)
            output = self.ffn_norm(attention_output + output)

        return output


class TransformerDecoderBlock(layers.Layer):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        inner_dim: int = None,
        output_dim: int = None,
        norm_first: bool = True,
        activation: Union[Callable, str] = "gelu",
        output_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        inner_dropout: float = 0.0,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        # ARCHITECTURE ATTRIBUTES
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim if inner_dim is not None else embed_dim
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self._norm_first = norm_first
        self.output_dropout = output_dropout
        self.attention_dropout = attention_dropout
        self.inner_dropout = inner_dropout
        self.epsilon = epsilon

        activation = keras.activations.get(activation)
        self.activation = activation

        # ARCHITECTURE LAYERS
        self.self_attention_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.attention_dropout,
            use_bias=True,
        )

        self.cross_attention_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.attention_dropout,
            use_bias=True,
        )

        inner_config = {
            "units": self.inner_dim,
            "activation": self.activation,
            "use_bias": True,
        }
        output_config = {
            "units": self.output_dim,
            "activation": self.activation,
            "use_bias": True,
        }
        self.ffn_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.ffn = keras.Sequential(
            [
                layers.Dense(**inner_config),
                layers.Dropout(rate=self.inner_dropout),
                layers.Dense(**output_config),
                layers.Dropout(rate=self.output_dropout),
            ]
        )

    def get_config(self):
        activation = keras.activations.serialize(self.activation)
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "inner_dim": self.inner_dim,
                "output_dim": self.output_dim,
                "norm_first": self._norm_first,
                "activation": activation,
                "output_dropout": self.output_dropout,
                "attention_dropout": self.attention_dropout,
                "inner_dropout": self.inner_dropout,
                "epsilon": self.epsilon,
            }
        )
        return config

    def call(self, inputs: DecoderBlockInput, training: bool) -> KerasTensor:
        # Inputs
        input_tensor, context_tensor, attention_mask, cross_attention_mask = (
            inputs["input"],
            inputs["context"],
            inputs.get("attention_mask", None),
            inputs.get("cross_attention_mask", None),
        )

        if self._norm_first:
            # Self attention block
            self_input_norm = self.self_attention_norm(input_tensor)
            self_output = self.self_attention(
                query=self_input_norm,
                value=self_input_norm,
                attention_mask=attention_mask,
                training=training,
            )
            self_output = input_tensor + self_output

            # Cross attention block
            cross_input_norm = self.cross_attention_norm(self_output)
            cross_output = self.cross_attention(
                query=cross_input_norm,
                value=context_tensor,
                attention_mask=cross_attention_mask,
                training=training,
            )
            cross_output = self_output + cross_output

            # Feed forward block
            output_norm = self.ffn_norm(cross_output)
            output = cross_output + self.ffn(output_norm)
        else:
            # Self attention block
            self_output = self.self_attention(
                query=input_tensor,
                value=input_tensor,
                attention_mask=attention_mask,
                training=training,
            )
            self_output = self.self_attention_norm(input_tensor + self_output)

            # Cross attention block
            cross_output = self.cross_attention(
                query=self_output,
                value=context_tensor,
                attention_mask=cross_attention_mask,
                training=training,
            )
            cross_output = self.cross_attention_norm(self_output + cross_output)

            # Feed forward block
            output = self.ffn(cross_output)
            output = self.ffn_norm(cross_output + output)

        return output
