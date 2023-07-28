"""Leverage FeatureSpace for abstract Structured Model"""


from typing import Any, Callable, Dict, Union

from typeguard import typechecked

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers, constraints


# TODO: finalize get_config + from_config


logger = tf.get_logger()
logger.setLevel("INFO")


class StructuredModel(keras.Model):
    """Structured Model based on FeatureSpace layer."""

    @typechecked
    def __init__(
        self,
        feature_space: keras.utils.FeatureSpace,
        embed_dim: int = 64,
        activation: Union[str, Callable] = "gelu",
        embed_mode: str = "concat",
        embeddings_initializer: str = "uniform",
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        embeddings_regularizer: Union[str, Callable] = None,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        embeddings_constraint: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        if not (feature_space.built and feature_space._is_adapted):
            raise ValueError(
                """`feature_space` is not built or adapted, 
                please provide a built and adapted feature_space argument.
                """
            )
        self.feature_space = feature_space

        if isinstance(activation, str):
            self.activation = keras.activations.get(activation)
        elif callable(activation):
            self.activation = activation
        else:
            raise TypeError(
                f"""`activation` must be a string or callable. 
                Received `{activation}`
                """
            )

        if embed_mode not in ["dict", "concat"]:
            raise ValueError(
                f"""Invalid value for argument `embed_mode`.
                Expected one of [`dict`, `concat`], received: output_mode={embed_mode}
                """
            )
        self.embed_mode = embed_mode

        # Weights initializers
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Weights regularizers
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Weights constraints
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Embedding layers
        for feature_name, preprocessor in self.feature_space.preprocessors.items():
            if isinstance(preprocessor, (layers.StringLookup, layers.IntegerLookup)):
                embed_layer = layers.Embedding(
                    input_dim=preprocessor.vocabulary_size(),
                    output_dim=self.embed_dim,
                    embeddings_initializer=self.embeddings_initializer,
                    embeddings_regularizer=self.embeddings_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    embeddings_constraint=self.embeddings_constraint,
                    name=f"{feature_name}_embedding",
                )
                logger.info(
                    f"Embedding layer instantiated for feature: `{feature_name}`"
                )
            elif isinstance(preprocessor, layers.Normalization):
                embed_layer = layers.Dense(
                    units=self.embed_dim,
                    activation=self.activation,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"{feature_name}_embedding",
                )
                logger.info(f"Dense layer instantiated for feature: `{feature_name}`")
            else:
                raise TypeError(
                    f"""Expected preprocessing layers are StringLookup, IntegerLookup and Normalization. 
                    Received: {type(preprocessor)}.
                    """
                )

            setattr(self, f"{feature_name}_embedding", embed_layer)
            logger.info(f"Embedding layer set for feature: `{feature_name}`")

        logger.info("Embedding layers built from `feature_space`.")

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        features = self.feature_space(inputs)

        if self.embed_mode == "concat":
            merged_features = []

            for feature_name in self.feature_space.features:
                embedded_feature = getattr(self, f"{feature_name}_embedding")(
                    features[feature_name]
                )
                if embedded_feature.shape.rank == 2:
                    embedded_feature = tf.expand_dims(embedded_feature, axis=1)

                merged_features.append(embedded_feature)

            features = tf.concat(merged_features, axis=1)

        else:
            merged_features = {}

            for feature_name in self.feature_space.features:
                embedded_feature = getattr(self, f"{feature_name}_embedding")(
                    features[feature_name]
                )
                if embedded_feature.shape.rank == 3:
                    embedded_feature = tf.squeeze(embedded_feature, axis=1)

                merged_features[feature_name] = embedded_feature

        return merged_features

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
