"""Self-Attention and Intersample Attention Transformer (SAINT) with TensorFlow"""

from typing import Callable, Dict, Union
import tensorflow as tf
from tensorflow.python.framework.tensor_util import ExtractBitsFromBFloat16

import schema
from .layers import MLP, SAINTBlock
from .augmentation import CutMix, Mixup


class SAINT(tf.keras.Model):
    """[summary]

    Args:
        input_schema (schema.InputFeaturesSchema): [description]
        n_layers (int): [description]
        num_heads (int): [description]
        embed_dim (int): [description]
        hidden_dim (int): [description]
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.
        cutmix_probability (float, optional): [description]. Defaults to 0.5.
        mixup_alpha (float, optional): [description]. Defaults to 0.5.
        seed (int, optional): [description]. Defaults to 26.
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
        n_layers: int,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        cutmix_probability: float = 0.5,
        mixup_alpha: float = 0.5,
        seed: int = 26,
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
        super(SAINT, self).__init__(**kwargs)

        # Input schema
        self.input_schema = input_schema

        # Transformer Encoder hyperparameters
        self.n_layers = n_layers

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.dropout = dropout
        self.epsilon = epsilon

        # Data augmentation hyperparameters
        self.cutmix_probability = cutmix_probability
        self.seed = seed
        self.mixup_alpha = mixup_alpha

        # Trainable weights
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.embeddings_regularizer = embeddings_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.embeddings_constraint = embeddings_constraint
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Self-supervised pretraining/supervised training
        self._pretraining = None

        self.set_inner_layers()

    def build(self):
        """Build CLS embedding, used for supervised downstream tasks."""

        self._CLS = self.add_weight(
            name="CLS",
            shape=(self.embed_dim,),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )

    def set_inner_layers(self):
        """Define SAINT layers."""

        # Data Augmentation layers
        self.cutmix_layer = CutMix(probability=self.probability, seed=self.seed)
        self.mixup_layer = Mixup(alpha=self.alpha, seed=self.seed)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Transformer Encoder
        self.SAINT = tf.keras.Sequential(
            [
                SAINTBlock(
                    num_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"SAINT_layer_{i}",
                )
                for i in range(self.n_layers)
            ]
        )

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
                setattr(
                    self,
                    f"{feature.name}_denoising",
                    MLP(
                        hidden_dim=self.embed_dim,
                        output_dim=feature.feature_dimension,
                        hidden_activation=tf.nn.relu,
                        output_activation=tf.nn.softmax,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        name=f"{feature.name}_denoising",
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
                setattr(
                    self,
                    f"{feature.name}_denoising",
                    MLP(
                        hidden_dim=self.embed_dim,
                        output_dim=feature.feature_dimension,
                        hidden_activation=tf.nn.relu,
                        output_activation=None,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        name=f"{feature.name}_denoising",
                    ),
                )

        # Projection head for input
        self.projection_head1 = MLP(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="projection_head1",
        )

        # Projection head for augmented input
        self.projection_head2 = MLP(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="projection_head2",
        )

    def EmbeddingLayer(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:

        # EMBED CATEGORICAL AND NUMERICAL FEATURES
        features_embeddings = [
            getattr(self, f"{feature.name}_embedding")(inputs[feature.name])
            for feature in self.input_schema.ordered_features
        ]

        return tf.stack(features_embeddings, axis=1)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:

        # FEATURES EMBEDDINGS LAYER
        features_embeddings = self.EmbeddingLayer(inputs)

        # CONCAT EMBEDDINGS
        key = next(iter(inputs))
        batch_size = tf.shape(inputs[key])[0]

        cls_embeddings = tf.tile(
            tf.reshape(self._CLS, (1, 1, self.embed_dim)),
            tf.constant([batch_size, 1, 1], dtype=tf.int32),
        )
        embeddings = tf.concat([cls_embeddings, features_embeddings], axis=1)

        # SAINT
        contextual_output = self.SAINT(embeddings, training)

        if not self._pretraining:  # CONTEXTUAL CLS
            return contextual_output[:, 0, :]

        return self.flatten(contextual_output)

    def compile(
        self,
        pretraining: bool,
        optimizer="adam",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        self._pretraining = pretraining
        super(SAINT, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs,
        )

    def train_step(self):
        raise NotImplementedError("Not yet implemented")

    def get_config(self):
        raise NotImplementedError("Not yet implemented")
