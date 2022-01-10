"""Self Attention and Intersample Attention blocks with TensorFlow"""

from typing import Callable, Dict, Union
import tensorflow as tf

from .schema import InputFeaturesSchema, FeatureType, input_schema_from_json


# TODO: add input_shape in Sequential model


def MLP(
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
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hidden_dim,
                activation=hidden_activation,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
            tf.keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
        ]
    )


class TabularEmbedding(tf.keras.layers.Layer):
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
        input_schema: InputFeaturesSchema,
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
        super(TabularEmbedding, self).__init__(**kwargs)

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

        self.set_inner_layers()

    def set_inner_layers(self):

        for feature in self.input_schema.ordered_features:
            if feature.feature_type is FeatureType.CATEGORICAL:
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
        config["input_schema"] = input_schema_from_json(config["input_schema"])
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
        base_config = super(TabularEmbedding, self).get_config()
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


class SelfAttentionBlock(tf.keras.layers.Layer):
    """[summary]

    Args:
        num_heads (int): [description]
        embed_dim (int): [description]
        hidden_dim (int): [description]
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        activity_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        super(SelfAttentionBlock, self).__init__(**kwargs)

        # Transformer/Attention block hyperparameters
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.epsilon = epsilon
        self.dropout = dropout

        # Trainable weights
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.set_inner_layers()

    def set_inner_layers(self):
        """Define Self Attention/Transformer block layers."""

        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.dropout,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.feed_forward_network = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            hidden_activation=tf.nn.gelu,
            output_activation=tf.nn.gelu,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]
            training (bool): [description]
            mask (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        attention_output = self.compute_attention(
            inputs, training=training, attention_mask=mask
        )
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layer_norm1(inputs + attention_output)

        output2 = self.feed_forward_network(output1)
        output2 = self.dropout2(output2, training=training)

        return self.layer_norm2(output1 + output2)

    def compute_attention(
        self, inputs: tf.Tensor, training: bool, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        return self.attention_layer(
            inputs, inputs, training=training, attention_mask=attention_mask
        )

    @classmethod
    def from_config(cls, config: Dict):
        config["kernel_initializer"] = tf.keras.initializers.deserialize(
            config["kernel_initializer"]
        )
        config["bias_initializer"] = tf.keras.initializers.deserialize(
            config["bias_initializer"]
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
        config["kernel_constraint"] = tf.keras.constraints.deserialize(
            config["kernel_constraint"]
        )
        config["bias_constraint"] = tf.keras.constraints.deserialize(
            config["bias_constraint"]
        )
        return cls(**config)

    def get_config(self) -> dict:
        base_config = super(SelfAttentionBlock, self).get_config()
        config = {
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "epsilon": self.epsilon,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": self.tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": self.tf.keras.regularizers.serialize(
                self.bias_regularizer
            ),
            "activity_regularizer": self.tf.keras.regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": self.tf.keras.constraints.serialize(
                self.kernel_constraint
            ),
            "bias_constraint": self.tf.keras.constraints.serialize(
                self.bias_constraint
            ),
        }
        return dict(list(base_config.items()) + list(config.items()))


class IntersampleAttentionBlock(SelfAttentionBlock):
    """[summary]

    Args:
        num_heads (int): [description]
        embed_dim (int): [description]
        hidden_dim (int): [description]
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        activity_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
    """

    def compute_attention(
        self, inputs: tf.Tensor, training: bool, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        """Intersample Attention operation.

        Args:
            inputs (tf.Tensor): [description]
            training (bool): [description]
            attention_mask (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        batch, n_samples, feature_dim = tf.shape(inputs)
        reshaped_inputs = tf.reshape(inputs, (1, batch, n_samples * feature_dim))

        attention_output = self.attention_layer(
            reshaped_inputs,
            reshaped_inputs,
            training=training,
            attention_mask=attention_mask,
        )
        output = tf.reshape(attention_output, (batch, n_samples, feature_dim))
        return output


def SAINTBlock(
    num_heads: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    epsilon: float = 1e-6,
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
        num_heads (int): [description]
        embed_dim (int): [description]
        hidden_dim (int): [description]
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.
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
    return tf.keras.Sequential(
        [
            SelfAttentionBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epsilon=epsilon,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
            IntersampleAttentionBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epsilon=epsilon,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
        ]
    )
