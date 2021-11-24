"""Self Attention and Intersample Attention blocks with TensorFlow"""

from typing import Callable, Union
import tensorflow as tf


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        bias_regularizer: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs
    ):
        """[summary]

        Args:
            embed_dim (int): [description]
            num_heads (int): [description]
            hidden_dim (int): [description]
            kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
            kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
            kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
            bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
            bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
            bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
            dropout (float, optional): [description]. Defaults to 0.1.
            epsilon (float, optional): [description]. Defaults to 1e-6.
        """
        super(SelfAttentionBlock, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.epsilon = epsilon
        self.dropout = dropout

        self.set_inner_layers()

    def set_inner_layers(self):
        """Define Self Attention/Transformer block layers."""

        self.Attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.dropout,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.FeedForwardNet = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.hidden_dim,
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                ),
                tf.keras.layers.Dense(
                    self.embed_dim,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                ),
            ]
        )

        self.LayerNorm1 = tf.keras.layers.LayerNormalization(self.epsilon)
        self.LayerNorm2 = tf.keras.layers.LayerNormalization(self.epsilon)

        self.Dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.Dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        """[summary]
        Args:
            inputs (tf.Tensor): [description]
            training (bool): [description]
            mask (TensorLike): [description]
        Returns:
            FloatTensorLike: [description]
        """
        attention_output = self.Attention(
            inputs, inputs, training=training, attention_mask=mask
        )
        attention_output = self.Dropout1(attention_output, training=training)
        output1 = self.LayerNorm1(inputs + attention_output)

        output2 = self.FeedForwardNet(output1)
        output2 = self.Dropout2(output2, training=training)

        return self.LayerNorm2(output1 + output2)

    def get_config(self):
        return super().get_config()


class IntersampleAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        bias_regularizer: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs
    ):
        """[summary]

        Args:
            embed_dim (int): [description]
            num_heads (int): [description]
            hidden_dim (int): [description]
            kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
            kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
            kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
            bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
            bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
            bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
            dropout (float, optional): [description]. Defaults to 0.1.
            epsilon (float, optional): [description]. Defaults to 1e-6.
        """
        super(IntersampleAttentionBlock, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.epsilon = epsilon
        self.dropout = dropout

        self.set_inner_layers()

    def set_inner_layers(self):
        """Define Intersample Attention block layers."""

        self.Attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.dropout,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.FeedForwardNet = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.hidden_dim,
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                ),
                tf.keras.layers.Dense(
                    self.embed_dim,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                ),
            ]
        )

        self.LayerNorm1 = tf.keras.layers.LayerNormalization(self.epsilon)
        self.LayerNorm2 = tf.keras.layers.LayerNormalization(self.epsilon)

        self.Dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.Dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        """[summary]
        Args:
            inputs (tf.Tensor): [description]
            training (bool): [description]
            mask (TensorLike): [description]
        Returns:
            FloatTensorLike: [description]
        """

        attention_output = self.intersample_attention(
            inputs, training=training, attention_mask=mask
        )
        attention_output = self.Dropout1(attention_output, training=training)
        output1 = self.LayerNorm1(inputs + attention_output)

        output2 = self.FeedForwardNet(output1)
        output2 = self.Dropout2(output2, training=training)

        return self.LayerNorm2(output1 + output2)

    def intersample_attention(
        self, inputs: tf.Tensor, training: bool, mask: tf.Tensor
    ) -> tf.Tensor:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]
            training (bool): [description]
            mask (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        batch, n_samples, feature_dim = tf.shape(inputs)
        reshaped_inputs = tf.reshape(inputs, (1, batch, n_samples * feature_dim))

        attention_output = self.Attention(
            reshaped_inputs, reshaped_inputs, training=training, attention_mask=mask
        )
        output = tf.reshape(attention_output, (batch, n_samples, feature_dim))
        return output

    def get_config(self):
        return super().get_config()


def SAINTBlock(
    embed_dim: int,
    num_heads: int,
    hidden_dim: int,
    kernel_initializer: Union[str, Callable] = "glorot_uniform",
    kernel_regularizer: Union[str, Callable] = None,
    kernel_constraint: Union[str, Callable] = None,
    bias_initializer: Union[str, Callable] = "zeros",
    bias_regularizer: Union[str, Callable] = None,
    bias_constraint: Union[str, Callable] = None,
    dropout: float = 0.1,
    epsilon: float = 1e-6,
    **kwargs
):
    """[summary]

    Args:
        name (str): [description]
        embed_dim (int): [description]
        num_heads (int): [description]
        hidden_dim (int): [description]
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.

    Returns:
        [type]: [description]
    """

    return tf.keras.Sequential(
        [
            SelfAttentionBlock(
                embed_dim,
                num_heads,
                hidden_dim,
                kernel_initializer,
                kernel_regularizer,
                kernel_constraint,
                bias_initializer,
                bias_regularizer,
                bias_constraint,
                dropout,
                epsilon,
                **kwargs
            ),
            IntersampleAttentionBlock(
                embed_dim,
                num_heads,
                hidden_dim,
                kernel_initializer,
                kernel_regularizer,
                kernel_constraint,
                bias_initializer,
                bias_regularizer,
                bias_constraint,
                dropout,
                epsilon,
                **kwargs
            ),
        ]
    )
