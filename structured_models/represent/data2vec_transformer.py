"""Data2vec Transformer for Product Representation"""


from typing import Callable, Dict, List, Tuple, Union
import tensorflow as tf

from ..structured_model import ImageAttributeModel
from ..layers import VanillaTransformerBlock, VisionTransformerBlock


logger = tf.get_logger()
logger.setLevel("INFO")


def Data2vecOptimizer() -> tf.keras.optimizers.Optimizer:
    raise NotImplementedError("Not implemented yet")


class Data2vecTransformer(ImageAttributeModel):
    """Multimodal self-supervised learning with data2vec.
    For more details on the learning procedure, see the paper:
    * https://arxiv.org/pdf/2202.03555.pdf.

    Example:
    ```python
    >>> import tensorflow as tf
    >>> import tensorflow_datasets as tfds
    >>> import tensorflow_data_validation as tfdv

    >>> from structured_transformers import Data2vecTransformer

    >>> model = Data2vecTransformer(
            num_blocks=6,
            num_heads=8,
            embed_dim=512,
            hidden_dim512,
            vision_transformer=True
        )
    ```

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
        embeddings_initializer (str, optional): _description_.
            Defaults to "uniform".
        kernel_initializer (Union[str, Callable], optional): _description_.
            Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): _description_.
            Defaults to "zeros".
        embeddings_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        activity_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        embeddings_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
    """

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        vision_transformer: bool = True,
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
        super().__init__(
            embed_dim=embed_dim,
            embeddings_initializer=embeddings_initializer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            embeddings_regularizer=embeddings_regularizer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epsilon = epsilon
        self.vision_transformer = vision_transformer

        self._top_blocks_target = None
        self._self_supervised_pretraining = None

    @property
    def pretraining(self):
        return self._self_supervised_pretraining

    @pretraining.setter
    def pretraining(self, value: bool = True):
        self._self_supervised_pretraining = value

    def build(self, input_shape: tf.TensorShape):
        # CLS TOKEN EMBEDDING
        self._CLS = self.add_weight(
            name="CLS",
            shape=(1, 1, self.embed_dim),
            initializer=self.embeddings_parameters["embeddings_initializer"],
            regularizer=self.embeddings_parameters["embeddings_regularizer"],
            constraint=self.embeddings_parameters["embeddings_constraint"],
        )

        # TEACHER & STUDENT TRANSFORMERS
        self._student_net = self._build_transformer(name="student")
        self._teacher_net = self._build_transformer(name="teacher")

        super().build(input_shape=input_shape)

    def _build_transformer(self, name: str) -> tf.keras.Model:
        """_summary_

        Args:
            block_fn (tf.keras.layers.Layer): _description_
            name (str): _description_

        Returns:
            tf.keras.Model: _description_
        """
        if self.vision_transformer:

            def block_fn(
                x: tf.Tensor, name: str, block_id: int
            ) -> tf.keras.layers.Layer:
                return VisionTransformerBlock(
                    num_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    ffn_output=True,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    name=name + "_vit_block" + str(block_id),
                    **self.weights_parameters,
                )(x)

        else:

            def block_fn(
                x: tf.Tensor, name: str, block_id: int
            ) -> tf.keras.layers.Layer:
                return VanillaTransformerBlock(
                    num_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    ffn_output=True,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    name=name + "_vanilla_block" + str(block_id),
                    **self.weights_parameters,
                )(x)

        inputs = tf.keras.Input(shape=(None, self.embed_dim), dtype=tf.float32)
        outputs: Dict[str, tf.Tensor] = {}

        x, ffn = block_fn(inputs, name, block_id=1)
        outputs[name + "_block1_ffn"] = ffn  # tf.reduce_mean(ffn, axis=1)

        for i in range(2, self.num_blocks + 1):
            x, ffn = block_fn(x, name, block_id=i)
            outputs[
                name + "_block" + str(i) + "_ffn"
            ] = ffn  # tf.reduce_mean(ffn, axis=1)

        outputs.update({"full_output": x, "cls_output": x[:, 0, :]})

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(
        self,
        top_blocks_target: int = 6,
        beta: float = 2.0,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        metrics: Union[str, tf.keras.metrics.Metric] = None,
        loss_weights: Union[List, Dict] = None,
        weighted_metrics: List[tf.keras.metrics.Metric] = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: bool = False,
        **kwargs,
    ):
        if top_blocks_target > self.num_blocks:
            logger.info(
                """`top_blocks_target` can't be strictly greather than `num_blocks`.
                `top_blocks_target` = min(`top_blocks_target`, `num_blocks`) is thus 
                considered.
                """
            )
        self._top_blocks_target = min(top_blocks_target, self.num_blocks)
        loss_obj = tf.keras.losses.Huber(delta=beta)

        super().compile(
            optimizer=optimizer,
            loss=loss_obj,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )

    def call(
        self,
        inputs: Union[Tuple[tf.Tensor], Dict[str, tf.Tensor]],
        training: bool,
        mask: tf.Tensor = None,
    ):
        raise NotImplementedError("Not yet implemented")

    def teacher_step(self):
        raise NotImplementedError("Not yet implemented")

    def train_step(self, data):
        raise NotImplementedError("Not yet implemented")

    def get_config(self):
        raise NotImplementedError("Not yet implemented")
