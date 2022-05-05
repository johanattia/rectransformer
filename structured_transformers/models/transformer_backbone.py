"""Self Attention and Intersample Attention blocks with TensorFlow"""

from typing import Callable, Dict, Union
import tensorflow as tf

from ..layers import TransformerBlock, IntersampleTransformerBlock


def TransformerEncoder(
    num_blocks: int,
    num_heads: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    epsilon: float = 1e-6,
    intersample_attention: bool = True,
    top_blocks_output: int = None,
    kernel_initializer: Union[str, Callable] = "glorot_uniform",
    bias_initializer: Union[str, Callable] = "zeros",
    kernel_regularizer: Union[str, Callable] = None,
    bias_regularizer: Union[str, Callable] = None,
    activity_regularizer: Union[str, Callable] = None,
    kernel_constraint: Union[str, Callable] = None,
    bias_constraint: Union[str, Callable] = None,
    **kwargs,
) -> tf.keras.Model:
    """Self-Attention and Intersample Attention Transformer (SAINT).

    Args:
        num_blocks (int): _description_
        num_heads (int): _description_
        embed_dim (int): _description_
        hidden_dim (int): _description_
        dropout (float, optional): _description_.
            Defaults to 0.1.
        epsilon (float, optional): _description_.
            Defaults to 1e-6.
        intersample_attention (bool, optional): _description_.
            Defaults to True.
        top_blocks_output (int, optional): _description_.
            Defaults to None.
        kernel_initializer (Union[str, Callable], optional): _description_.
            Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): _description_.
            Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        activity_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.

    Returns:
        tf.keras.Model: _description_
    """
    # Parameters
    weights_parameters = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )
    name = kwargs.pop("name", "transformer")

    # Define Transformer block
    if intersample_attention:

        def block_fn(x, block_idx: int):
            x = TransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epsilon=epsilon,
                name=name + "_transformer_block" + str(block_idx),
                **weights_parameters,
                **kwargs,
            )(x)
            x = IntersampleTransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epsilon=epsilon,
                name=name + "_intersample_block" + str(block_idx),
                **weights_parameters,
                **kwargs,
            )(x)
            return x

    else:

        def block_fn(x, block_idx: int):
            return TransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epsilon=epsilon,
                name=name + "_transformer_block" + str(block_idx),
                **weights_parameters,
                **kwargs,
            )(x)

    # Transformer Encoder
    x = tf.keras.Input(shape=(None, embed_dim), dtype=tf.float32)
    output: Dict[str, tf.Tensor] = {}

    for idx in range(1, num_blocks + 1):
        x = block_fn(x, block_idx=idx)

        if isinstance(top_blocks_output is not None) and (
            idx >= num_blocks - top_blocks_output
        ):
            output[name + "_block" + str(idx)] = x
        elif (top_blocks_output is None) and (idx == num_blocks):
            output = {
                "full_output": x,
                "cls_output": x[:, 0, :],
            }

    return tf.keras.Model(inputs=x, outputs=output, name=name)
