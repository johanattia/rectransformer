"""Transformer encoder with TensorFlow"""

from typing import Callable, Dict, Tuple, Union
import tensorflow as tf

from ..layers import VanillaTransformerBlock, VisionTransformerBlock


def TransformerEncoder(
    num_blocks: int,
    num_heads: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    epsilon: float = 1e-6,
    vision_transformer: bool = True,
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
    """Instantiate Vanilla/Vision Transformer encoder.

    Args:
        num_blocks (int): _description_
        num_heads (int): _description_
        embed_dim (int): _description_
        hidden_dim (int): _description_
        dropout (float, optional): _description_.
            Defaults to 0.1.
        epsilon (float, optional): _description_.
            Defaults to 1e-6.
        vision_transformer (bool, optional): _description_.
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
    if top_blocks_output is not None:
        if num_blocks == 1:
            raise ValueError(
                """It doesn't make sense to extract `ffn_output` with `num_blocks` == 1. 
                Please consider `num_blocks` > 1 then `top_blocks_output` at most equal 
                to `num_blocks`.
                """
            )
        elif top_blocks_output > num_blocks:
            raise ValueError(
                "Please select `top_blocks_output` at most equal to `num_blocks`."
            )
        else:
            ffn_output = True
            block_begin = num_blocks - top_blocks_output + 1
    else:
        ffn_output = False

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
    name = kwargs.pop("name", "encoder")

    # Define Transformer block
    if vision_transformer:

        def block_fn(x: tf.Tensor, block_id: int) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
            return VisionTransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                ffn_output=ffn_output,
                dropout=dropout,
                epsilon=epsilon,
                name=name + "_vit_block" + str(block_id),
                **weights_parameters,
                **kwargs,
            )(x)

    else:

        def block_fn(x: tf.Tensor, block_id: int) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
            return VanillaTransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                ffn_output=ffn_output,
                dropout=dropout,
                epsilon=epsilon,
                name=name + "_transformer_block" + str(block_id),
                **weights_parameters,
                **kwargs,
            )(x)

    # Transformer Encoder
    inputs = tf.keras.Input(shape=(None, embed_dim), dtype=tf.float32)
    outputs: Dict[str, tf.Tensor] = {}

    if ffn_output:
        x, ffn = block_fn(inputs, block_id=1)

        if block_begin >= 1:
            outputs[name + "_block1_ffn"] = ffn
    else:
        x = block_fn(inputs, block_id=1)

    for idx in range(2, num_blocks + 1):
        if ffn_output:
            x, ffn = block_fn(x, block_id=idx)

            if block_begin >= idx:
                outputs[name + "_block" + str(idx) + "_ffn"] = ffn
        else:
            x = block_fn(x, block_id=idx)

    outputs["full_output"] = x
    outputs["cls_output"] = x[:, 0, :]

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
