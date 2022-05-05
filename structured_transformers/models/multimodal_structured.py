"""Multimodal Image Attribute Model and schema integration with TensorFlow"""

import tensorflow as tf


class ImageAttributeModel(tf.keras.Model):
    def __init__(
        self,
    ):
        raise NotImplementedError

    def build_from_schema_and_dataset(
        self,
    ):
        raise NotImplementedError

    def apply_preprocessing(
        self,
    ):
        raise NotImplementedError
