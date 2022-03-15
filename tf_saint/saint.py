"""Self-Attention and Intersample Attention Transformer (SAINT) with TensorFlow"""

from typing import Callable, Dict, Iterable, Union

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from .schema import FeatureType, InputFeaturesSchema
from .layers import MLP, StructuredEmbedding, SAINTBlock
from .augmentation import CutMix, Mixup


class SAINT(tf.keras.Model):
    """[summary]

    Args:
        input_schema (InputFeaturesSchema): [description]
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
        input_schema: InputFeaturesSchema,
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
        super().__init__(**kwargs)

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

    def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
        # CLS token
        self._CLS = self.add_weight(
            name="CLS",
            shape=(1, 1, self.embed_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )

        # Data Augmentation layers
        self.cutmix = CutMix(probability=self.probability, seed=self.seed)
        self.mixup = Mixup(alpha=self.alpha, seed=self.seed)

        # Tabular Embedding layer
        self.embedding = StructuredEmbedding(
            input_schema=self.input_schema,
            embed_dim=self.embed_dim,
            embeddings_initializer=self.embeddings_initializer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            embeddings_regularizer=self.embeddings_regularizer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            embeddings_constraint=self.embeddings_constraint,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        # Transformer Encoder
        self.saint = tf.keras.Sequential(
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

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Denoising layers
        for feature in self.input_schema.ordered_features:
            setattr(
                self,
                f"{feature.name}_denoising",
                MLP(
                    hidden_dim=self.embed_dim,
                    output_dim=feature.feature_dimension,
                    hidden_activation=tf.nn.relu,
                    output_activation=tf.nn.softmax
                    if feature.feature_type is FeatureType.CATEGORICAL
                    else None,
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

        super().build(input_shape)

    def call(
        self,
        inputs: Union[tf.Tensor, Dict[str, tf.Tensor]],
        training: bool,
        augmentation: bool = False,
    ) -> tf.Tensor:
        # Structured Embedding layer
        if augmentation:
            augmented_inputs = self.cutmix(inputs)
            features_embeddings = self.mixup(self.embedding(augmented_inputs))
        else:
            features_embeddings = self.embedding(inputs)

        # Concat embeddings
        batch_size = tf.shape(features_embeddings)[0]
        cls_embeddings = tf.repeat(
            self._CLS, repeats=tf.constant([batch_size], dtype=tf.int32), axis=0
        )
        embeddings = tf.concat([cls_embeddings, features_embeddings], axis=1)

        # SAINT
        contextual_output = self.saint(embeddings, training)

        # Output
        batch_output = {
            "output": contextual_output,
            "flatten_output": self.flatten(contextual_output),
            "cls_output": contextual_output[:, 0, :]
        }

        return batch_output

    def compile(
        self,
        contrastive_temperature: float,
        contrastive_optimizer: tf.keras.optimizers.Optimizer,
        reconstruction_optimizer: tf.keras.optimizers.Optimizer,
        **kwargs
    ):
        """
        
        """
        super().compile(**kwargs)

        self.contrastive_temperature = contrastive_temperature
        self.contrastive_optimizer = contrastive_optimizer
        self.reconstruction_optimizer = reconstruction_optimizer
        
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss_tracker")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="contrastive_accuracy"
        )

        self.reconstruction_losses = {}

    def train_step(self, data: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        # Processing data before forward
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        # Contrastive learning
        with tf.GradientTape() as tape:
            features = self(x, training=True)
            augmented_features = self(x, training=True, augmentation=True)

            flatten_output = features["flatten_output"]
            augmented_flatten_output = augmented_features["flatten_output"]

        # Contrastive learning

        return NotImplemented

    def contrastive_loss(self, features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:

        normalized_features1 = tf.math.l2_normalize(projections_1, axis=1)
        normalized_features2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
            tf.matmul(
                normalized_features1,
                normalized_features2,
                transpose_b=True
            ) / self.contrastive_temperature
        )

        batch_size = tf.shape(features1)[0]
        contrastive_labels = tf.range(batch_size)
        
        return

    def get_config(self):
        raise NotImplementedError("Not yet implemented")
