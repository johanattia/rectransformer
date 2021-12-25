"""Self-Attention and Intersample Attention Transformer (SAINT) with TensorFlow"""

from typing import Callable, Dict, Iterable, Union
import tensorflow as tf

from .layers import MLP, SAINTBlock
from .augmentation import CutMixLayer, MixupLayer


class SAINT(tf.keras.Model):
    """[summary]

    Args:
        n_layers (int): [description]
        categorical_variables (Dict[str, int]): [description]
        numerical_variables (Iterable[str]): [description]
        embed_dim (int): [description]
        num_heads (int): [description]
        hidden_dim (int): [description]
        embeddings_initializer (str, optional): [description]. Defaults to "uniform".
        embeddings_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        embeddings_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_initializer (Union[str, Callable], optional): [description]. Defaults to "glorot_uniform".
        kernel_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        bias_initializer (Union[str, Callable], optional): [description]. Defaults to "zeros".
        bias_regularizer (Union[str, Callable], optional): [description]. Defaults to None.
        bias_constraint (Union[str, Callable], optional): [description]. Defaults to None.
        dropout (float, optional): [description]. Defaults to 0.1.
        epsilon (float, optional): [description]. Defaults to 1e-6.
    """

    def __init__(
        self,
        n_layers: int,
        categorical_variables: Dict[str, int],
        numerical_variables: Iterable[str],
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        embeddings_initializer: str = "uniform",
        embeddings_regularizer: Union[str, Callable] = None,
        embeddings_constraint: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        bias_regularizer: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super(SAINT, self).__init__(**kwargs)

        self.n_layers = n_layers

        self.categorical_variables = categorical_variables
        self.numerical_variables = numerical_variables

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.epsilon = epsilon
        self.dropout = dropout

        self.set_inner_layers()

    def set_inner_layers(self):

        # SAINT Transformer Encoder layers
        self.encoder = tf.keras.Sequential(
            [
                SAINTBlock(
                    self.embed_dim,
                    self.num_heads,
                    self.hidden_dim,
                    self.kernel_initializer,
                    self.kernel_regularizer,
                    self.kernel_constraint,
                    self.bias_initializer,
                    self.bias_regularizer,
                    self.bias_constraint,
                    self.dropout,
                    self.epsilon,
                    name=f"SAINT_layer_{i}",
                )
                for i in range(self.n_layers)
            ]
        )

        # Embedding and denoising MLPs for categorical variables
        for var_name, var_dimension in self.categorical_variables.items():
            setattr(
                self,
                f"{var_name}_embedding",
                tf.keras.layers.Embedding(
                    var_dimension,
                    self.embed_dim,
                    self.embeddings_initializer,
                    self.embeddings_regularizer,
                    self.embeddings_constraint,
                    name=f"{var_name}_embedding",
                ),
            )
            setattr(
                self,
                f"{var_name}_denoising",
                MLP(
                    hidden_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    output_activation=tf.nn.softmax,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"{var_name}_dense",
                ),
            )

        # Embedding and denoising MLP for numerical variables
        for var_name in self.numerical_variables:
            setattr(
                self,
                f"{var_name}_dense",
                tf.keras.layers.Dense(
                    self.embed_dim,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"{var_name}_dense",
                ),
            )
            setattr(
                self,
                f"{var_name}_denoising",
                MLP(
                    hidden_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"{var_name}_dense",
                ),
            )

        # Projection head for input
        self.projection_head1 = MLP(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"projection_head1",
        )

        # Projection head for augmented input
        self.projection_head2 = MLP(
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"projection_head2",
        )

    def compile(self):
        raise NotImplementedError("Not yet implemented")

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        raise NotImplementedError("Not yet implemented")

    def train_step(self):
        raise NotImplementedError("Not yet implemented")

    def get_config(self):
        raise NotImplementedError("Not yet implemented")
