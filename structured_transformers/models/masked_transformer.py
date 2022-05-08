"""Masked Auto-encoder model for self-supervised learning with TensorFlow"""

from typing import Callable, Dict, Iterable, Union

import tensorflow as tf

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

from .transformer import TransformerEncoder

from ..layers import CutMix, Mixup
from ..layers import FeedForwardNetwork


schema_utils = tfdv.utils.schema_util


# TODO:
# Explore for optimization : tf.nest package + tf.unstack


class SAINT(tf.keras.Model):
    """Self-Attention and Intersample Attention Transformer (SAINT).
    The training logic implemented is a self-supervised pre-training associated to
    a features reconstruction task.

    For more details, see the paper: https://arxiv.org/pdf/2106.01342.pdf.

    Example:
    ```python
    >>> import tensorflow as tf
    >>> import tensorflow_addons as tfa
    >>> import tensorflow_datasets as tfds
    >>> import tensorflow_data_validation as tfdv

    >>> import structured_transformers

    >>> diamonds_ds = tfds.load('diamonds', split='train', shuffle_files=True)
    >>> diamonds_df = tfds.as_dataframe(diamonds_ds)

    >>> diamonds_stats = tfdv.generate_statistics_from_dataframe(diamonds_df)
    >>> diamonds_schema = tfdv.infer_schema(statistics=diamonds_stats)

    >>> model = structured_transformers.models.SAINT(
            num_blocks=6,
            num_heads=8,
            embed_dim=512,
            hidden_dim512,
        )
    >>> model.build_from_schema_and_dataset(diamonds_schema, diamonds_ds)

    >>> diamonds_ds = diamonds_ds.shuffle(buffer_size=10000).batch(batch_size=512)
    >>> diamonds_ds = model.apply_preprocessing(diamonds_ds, as_supervised=False)

    >>> model.compile(
            optimizer=tfa.optimizers.LAMB(),
            pretraining_loss=structured_transformers.losses.VICReg(),
            denoising_lambda=7.0,
        )
    >>> history = model.fit(diamonds_ds, epochs=20)
    ```
    """

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        intersample_attention: bool = True,
        top_blocks_output: int = None,
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
        """Instantiate Self-Attention and Intersample Attention Transformer with
        self-supervised pre-training procedure.

        Args:
            num_blocks (int): _description_
            num_heads (int): _description_
            embed_dim (int): _description_
            hidden_dim (int): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
            epsilon (float, optional): _description_. Defaults to 1e-6.
            intersample_attention (bool, optional): _description_. Defaults to True.
            top_blocks_output (int, optional): _description_. Defaults to None.
            cutmix_probability (float, optional): _description_. Defaults to 0.5.
            mixup_alpha (float, optional): _description_. Defaults to 0.5.
            seed (int, optional): _description_. Defaults to 26.
            embeddings_initializer (str, optional): _description_. Defaults to "uniform".
            kernel_initializer (Union[str, Callable], optional): _description_. Defaults to "glorot_uniform".
            bias_initializer (Union[str, Callable], optional): _description_. Defaults to "zeros".
            embeddings_regularizer (Union[str, Callable], optional): _description_. Defaults to None.
            kernel_regularizer (Union[str, Callable], optional): _description_. Defaults to None.
            bias_regularizer (Union[str, Callable], optional): _description_. Defaults to None.
            activity_regularizer (Union[str, Callable], optional): _description_. Defaults to None.
            embeddings_constraint (Union[str, Callable], optional): _description_. Defaults to None.
            kernel_constraint (Union[str, Callable], optional): _description_. Defaults to None.
            bias_constraint (Union[str, Callable], optional): _description_. Defaults to None.
        """
        super().__init__(**kwargs)

        # Transformer Encoder hyperparameters
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epsilon = epsilon
        self.intersample_attention = intersample_attention
        self.top_blocks_output = top_blocks_output

        # Data augmentation hyperparameters
        self.cutmix_probability = cutmix_probability
        self.mixup_alpha = mixup_alpha
        self.seed = seed

        # Trainable weights
        self.embeddings_parameters = dict(
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
        )
        self.weights_parameters = dict(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )

        # Data schema
        self._schema = None
        self._is_built = None

    def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
        """_summary_

        Args:
            input_shape (Union[tf.TensorShape, Iterable[tf.TensorShape]]): _description_
        """
        # CLS token
        self._CLS = self.add_weight(
            name="CLS",
            shape=(1, 1, self.embed_dim),
            initializer=self.embeddings_parameters["embeddings_initializer"],
            regularizer=self.embeddings_parameters["embeddings_regularizer"],
            constraint=self.embeddings_parameters["embeddings_constraint"],
        )

        # Non-trainable layers
        self.flatten = tf.keras.layers.Flatten()
        self.cutmix = CutMix(probability=self.cutmix_probability, seed=self.seed)
        self.mixup = Mixup(alpha=self.mixup_alpha, seed=self.seed)

        # Trainable layers
        # Transformer encoder
        self.saint = TransformerEncoder(
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            epsilon=self.epsilon,
            intersample_attention=self.intersample_attention,
            top_blocks_output=self.top_blocks_output,
            **self.weights_parameters,
        )
        # Projection head for input
        self.projection_head1 = FeedForwardNetwork(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            **self.weights_parameters,
            name="projection_head1",
        )
        # Projection head for augmented input
        self.projection_head2 = FeedForwardNetwork(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            **self.weights_parameters,
            name="projection_head2",
        )

        # Inheritance
        super().build(input_shape)

    def build_from_schema_and_dataset(
        self,
        schema: schema_pb2.Schema,
        dataset: tf.data.Dataset,
        as_supervised: bool = False,
    ) -> tf.data.Dataset:
        """Perform preprocessing and build embeddings layers from protobuf schema and
        training dataset.

        Args:
            schema (schema_pb2.Schema): Protocol buffers schema of training data.
            dataset (tf.data.Dataset): Training dataset. Samples should be formated as dict
                following the protobuf data schema.
            as_supervised (bool): whether the input `dataset` has a 2-tuple structure
                `(inputs, labels)` (True) or an only-features dict structure (False).
                Defaults to False.

        Returns:
            tf.data.Dataset: _description_
        """
        for feature in schema.feature:
            if as_supervised:
                feature_dataset = dataset.map(lambda x, y: x[feature.name])
            else:
                feature_dataset = dataset.map(lambda x: x[feature.name])

            if feature.type == schema_pb2.FLOAT:
                preprocessing_layer = tf.keras.layers.Normalization()
                preprocessing_layer.adapt(feature_dataset)
                output_dim = 1

                setattr(
                    self,
                    f"{feature.name}_embedding",
                    tf.keras.layers.Dense(
                        units=self.embed_dim,
                        activation=tf.nn.relu,
                        use_bias=True,
                        **self.weights_parameters,
                        name=f"{feature.name}_embedding",
                    ),
                )
            else:
                if feature.type == schema_pb2.INT:
                    feature.int_domain.is_categorical = True
                    preprocessing_layer = tf.keras.layers.IntegerLookup()
                elif feature.type == schema_pb2.BYTES:
                    preprocessing_layer = tf.keras.layers.StringLookup()

                preprocessing_layer.adapt(feature_dataset)
                output_dim = preprocessing_layer.vocabulary_size()

                setattr(
                    self,
                    f"{feature.name}_embedding",
                    tf.keras.layers.Embedding(
                        input_dim=preprocessing_layer.vocabulary_size(),
                        output_dim=self.embed_dim,
                        **self.embeddings_parameters,
                        name=f"{feature.name}_embedding",
                    ),
                )

            setattr(self, f"{feature.name}_preprocessing", preprocessing_layer)
            setattr(
                self,
                f"{feature.name}_denoising",
                FeedForwardNetwork(
                    hidden_dim=self.embed_dim,
                    output_dim=output_dim,
                    hidden_activation=tf.nn.relu,
                    **self.weights_parameters,
                    name=f"{feature.name}_denoising",
                ),
            )

        self._is_built = True
        self._schema = schema

        dataset = self.apply_preprocessing(dataset, as_supervised)
        return dataset

    def apply_preprocessing(
        self, dataset: tf.data.Dataset, as_supervised: bool = False
    ) -> tf.data.Dataset:
        """Apply preprocessing layers at inference/prediction step.

        Args:
            dataset (tf.data.Dataset): Prediction dataset.
            as_supervised (bool): whether the input `dataset` has a 2-tuple structure
                `(inputs, labels)` (True) or an only-features dict structure (False).
                Defaults to False.

        Returns:
            tf.data.Dataset: Preprocessed dataset, ready for prediction.
        """
        if self._schema is None:
            raise AttributeError(
                """
                A valid protobuf schema must be given at a previous step, e.g. using
                `build_from_schema_and_dataset` method. For more details, see source code
                or documentation.
                """
            )
        if as_supervised:
            preprocess_fn = lambda x, y: (
                {
                    feature.name: getattr(self, f"{feature.name}_preprocessing")(
                        x[feature.name]
                    )
                    for feature in self._schema.feature
                },
                y,
            )
        else:
            preprocess_fn = lambda x: {
                feature.name: getattr(self, f"{feature.name}_preprocessing")(
                    x[feature.name]
                )
                for feature in self._schema.feature
            }

        dataset = dataset.map(preprocess_fn)
        return dataset

    def call(
        self,
        inputs: Union[tf.Tensor, Dict[str, tf.Tensor]],
        training: bool,
        augment: bool = False,
    ) -> tf.Tensor:
        """_summary_

        Args:
            inputs (Union[tf.Tensor, Dict[str, tf.Tensor]]): _description_
            training (bool): _description_
            augmentation (bool, optional): _description_. Defaults to False.

        Returns:
            tf.Tensor: _description_
        """
        # Embedding layer
        if augment:
            augmented_inputs = self.cutmix(inputs)
            features_embeddings = tf.stack(
                [
                    getattr(self, f"{feature.name}_embedding")(
                        augmented_inputs[feature.name]
                    )
                    for feature in self._schema.feature
                ],
                axis=1,
            )
            features_embeddings = self.mixup(features_embeddings)
        else:
            features_embeddings = tf.stack(
                [
                    getattr(self, f"{feature.name}_embedding")(inputs[feature.name])
                    for feature in self._schema.feature
                ],
                axis=1,
            )

        # Concat embeddings
        batch_size = tf.shape(features_embeddings)[0]
        cls_embeddings = tf.repeat(
            self._CLS, repeats=tf.constant([batch_size], dtype=tf.int32), axis=0
        )
        embeddings = tf.concat([cls_embeddings, features_embeddings], axis=1)

        # SAINT
        output = self.saint(embeddings, training)
        return output

    def compile(
        self, pretraining_loss: tf.keras.losses.Loss, denoising_lambda: float, **kwargs
    ):
        """_summary_

        Args:
            pretraining_loss (tf.keras.losses.Loss): _description_
            denoising_lambda (float): _description_
        """
        super().compile(**kwargs)

        self.pretraining_loss = pretraining_loss
        self.denoising_lambda = denoising_lambda

        self.pretraining_loss_loss_tracker = tf.keras.metrics.Mean(
            name="pretraining_loss_loss_tracker"
        )
        self.denoising_tracker = {
            feature.name: tf.keras.metrics.Mean(name=f"{feature.name}_tracker")
            for feature in self._schema.feature
        }

    def train_step(self, data: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        """Training logic with two losses to optimize:
        - InfoNCE loss for contrastive learning, including data augmentation
        - Denoising loss for features reconstruction

        Args:
            data (Union[tf.Tensor, Dict[str, tf.Tensor]]): _description_

        Returns:
            _type_: _description_
        """
        # Processing data before forward propagation
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            # Forward
            output1 = self(x, training=True)  # augment=True
            output1 = self.flatten(output1["full_output"])

            output2 = self(x, training=True, augment=True)
            output2 = self.flatten(output2["full_output"])

            # Contrastive loss
            projection1 = self.projection_head1(output1)
            projection2 = self.projection_head2(output2)

            pretraining_loss = self.pretraining_loss(
                projection1=projection1, projection2=projection2
            )

            # Denoising losses
            denoising_outputs = {
                feature.name: getattr(self, f"{feature.name}_denoising")(output2)
                for feature in self._schema.feature
            }
            denoising_loss = self.denoising_loss(inputs=x, outputs=denoising_outputs)

            # Pre-training loss
            loss = pretraining_loss + self.denoising_lambda * denoising_loss

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return NotImplemented

    def contrastive_loss(
        self, projection1: tf.Tensor, projection2: tf.Tensor
    ) -> tf.Tensor:
        """Contrastive loss implementation from https://keras.io/examples/vision/semisupervised_simclr/.

        Args:
            projection1 (tf.Tensor): _description_
            projection2 (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        normalized_projection1 = tf.math.l2_normalize(projection1, axis=1)
        normalized_projection2 = tf.math.l2_normalize(projection2, axis=1)

        similarities = (
            tf.matmul(normalized_projection1, normalized_projection2, transpose_b=True)
            / self.contrastive_temperature
        )

        batch_size = tf.shape(projection1)[0]
        contrastive_labels = tf.range(batch_size)

        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def denoising_loss(
        self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """_summary_

        Args:
            inputs (Dict[str, tf.Tensor]): _description_
            outputs (Dict[str, tf.Tensor]): _description_

        Returns:
            tf.Tensor: _description_
        """
        loss = 0
        for feature in self._schema.feature:

            if schema_utils.is_categorical_feature(feature):
                feature_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    inputs[feature.name], outputs[feature.name], from_logits=True
                )
            else:
                feature_loss = tf.keras.losses.mean_squared_error(
                    inputs[feature.name], outputs[feature.name]
                )

            self.denoising_tracker[feature.name].update_state(feature_loss)
            loss += feature_loss

        return loss

    def get_config(self):
        raise NotImplementedError("Not yet implemented")
