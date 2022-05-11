"""Baskets Transformer for Supervised Sequence-of-baskets Modelling"""


import tensorflow as tf

from ..layers import VanillaTransformerBlock


class BasketsTransformer(tf.keras.Model):
    """_summary_

    References:
    * Generating Long Sequences with Sparse Transformers: https://arxiv.org/pdf/1904.10509.pdf
    * DeepNet: Scaling Transformers to 1,000 Layers: https://arxiv.org/pdf/2203.00555.pdf

    Args:
        tf (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.product_embedding =
        # self.position_encoding =

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        return super().get_config()
