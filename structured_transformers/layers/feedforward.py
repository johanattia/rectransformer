"""Self Attention and Intersample Attention blocks with TensorFlow"""

from typing import Callable, Union
import tensorflow as tf


def FeedForwardNetwork(
    hidden_dim: int,
    output_dim: int,
    hidden_activation: Union[str, Callable] = None,
    output_activation: Union[str, Callable] = None,
    kernel_initializer: Union[str, Callable] = "glorot_uniform",
    bias_initializer: Union[str, Callable] = "zeros",
    kernel_regularizer: Union[str, Callable] = None,
    bias_regularizer: Union[str, Callable] = None,
    activity_regularizer: Union[str, Callable] = None,
    kernel_constraint: Union[str, Callable] = None,
    bias_constraint: Union[str, Callable] = None,
    **kwargs,
) -> tf.keras.Model:
    """[summary]

    Args:
        hidden_dim (int): [description]
        output_dim (int): [description]
        hidden_activation (Union[str, Callable], optional): [description]. Defaults to None.
        output_activation (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        activity_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.

    Returns:
        tf.keras.Model: [description]
    """
    weights_parameters = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hidden_dim,
                activation=hidden_activation,
                use_bias=True,
                **weights_parameters,
                **kwargs,
            ),
            tf.keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                use_bias=True,
                **weights_parameters,
                **kwargs,
            ),
        ]
    )
