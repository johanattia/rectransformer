"""Embedding layers with TensorFlow"""

from typing import Callable, Dict, Iterable, Union
import tensorflow as tf

from ..utils import schema


class StructuredEmbedding(tf.keras.layers.Layer):
    """[summary]

    Args:
        input_schema (schema.InputFeaturesSchema): [description]
        embed_dim (int): [description]
        embeddings_initializer (str, optional): [description]. Defaults to "uniform".
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        embeddings_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        activity_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        embeddings_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        input_schema: schema.InputFeaturesSchema,
        embed_dim: int,
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
        super(StructuredEmbedding, self).__init__(**kwargs)

        # Input schema
        self.input_schema = input_schema

        # Embedding dim
        self.embed_dim = embed_dim

        # Trainable weights
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
        # Ddefining numerical/categorical embedding layers
        for feature in self.input_schema.ordered_features:
            if feature.feature_type is schema.FeatureType.CATEGORICAL:
                setattr(
                    self,
                    f"{feature.name}_embedding",
                    tf.keras.layers.Embedding(
                        input_dim=feature.feature_dimension,
                        output_dim=self.embed_dim,
                        embeddings_initializer=self.embeddings_initializer,
                        embeddings_regularizer=self.embeddings_regularizer,
                        embeddings_constraint=self.embeddings_constraint,
                        name=f"{feature.name}_embedding",
                    ),
                )
            else:
                setattr(
                    self,
                    f"{feature.name}_embedding",
                    tf.keras.layers.Dense(
                        units=self.embed_dim,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        name=f"{feature.name}_embedding",
                    ),
                )

        super(StructuredEmbedding, self).build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        return tf.stack(
            [
                getattr(self, f"{feature.name}_embedding")(inputs[feature.name])
                for feature in self.input_schema.ordered_features
            ],
            axis=1,
        )

    @classmethod
    def from_config(cls, config: Dict):
        config["input_schema"] = schema.input_schema_from_json(config["input_schema"])
        config["embeddings_initializer"] = tf.keras.initializers.deserialize(
            config["embeddings_initializer"]
        )
        config["kernel_initializer"] = tf.keras.initializers.deserialize(
            config["kernel_initializer"]
        )
        config["bias_initializer"] = tf.keras.initializers.deserialize(
            config["bias_initializer"]
        )
        config["embeddings_regularizer"] = tf.keras.regularizers.deserialize(
            config["embeddings_regularizer"]
        )
        config["kernel_regularizer"] = tf.keras.regularizers.deserialize(
            config["kernel_regularizer"]
        )
        config["bias_regularizer"] = tf.keras.regularizers.deserialize(
            config["bias_regularizer"]
        )
        config["activity_regularizer"] = tf.keras.regularizers.deserialize(
            config["activity_regularizer"]
        )
        config["embeddings_constraint"] = tf.keras.constraints.deserialize(
            config["embeddings_constraint"]
        )
        config["kernel_constraint"] = tf.keras.constraints.deserialize(
            config["kernel_constraint"]
        )
        config["bias_constraint"] = tf.keras.constraints.deserialize(
            config["bias_constraint"]
        )
        return cls(**config)

    def get_config(self) -> dict:
        base_config = super(StructuredEmbedding, self).get_config()
        config = {
            "input_schema": self.input_schema.json(),
            "embed_dim": self.embed_dim,
            "embeddings_initializer": tf.keras.initializers.serialize(
                self.embeddings_initializer
            ),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "embeddings_regularizer": tf.keras.regularizers.serialize(
                self.embeddings_regularizer
            ),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(
                self.activity_regularizer
            ),
            "embeddings_constraint": tf.keras.constraints.serialize(
                self.embeddings_constraint
            ),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
        }
        return dict(list(base_config.items()) + list(config.items()))
