from enum import Enum
from typing import Callable, Dict, Union

from dataclasses import asdict
from pydantic.dataclasses import dataclass

import keras
from keras import layers

import torch
from torch import nn


class FeatureArgumentError(Exception):
    pass


class FeatureMode(Enum):
    FLOAT = "float"
    INT = "int"
    ONE_HOT = "one_hot"


class FeatureLayer(Enum):
    DENSE = "dense"
    EMBEDDING = "embedding"
    IDENTITY = "identity"


@dataclass
class FeatureConfig:
    """Dataclass to store needed information about features."""

    name: str
    mode: FeatureMode
    cardinality: int
    layer: FeatureLayer

    def to_layer(
        self,
        output_dim: int,
        activation: Union[str, Callable] = None,
        use_bias: bool = True,
        embeddings_initializer: Union[str, Callable] = "uniform",
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        embeddings_regularizer: Union[str, Callable] = None,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        embeddings_constraint: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        mask_zero=False,
        suffix: str = "",
    ) -> layers.Layer:
        base_name = "_".join([self.name, suffix]) if suffix != "" else self.name

        if self.layer == FeatureLayer.DENSE and self.mode in [
            FeatureMode.FLOAT,
            FeatureMode.ONE_HOT,
        ]:
            layer_config = {
                "units": output_dim,
                "activation": activation,
                "use_bias": use_bias,
                "kernel_initializer": kernel_initializer,
                "bias_initializer": bias_initializer,
                "kernel_regularizer": kernel_regularizer,
                "bias_regularizer": bias_regularizer,
                "activity_regularizer": activity_regularizer,
                "kernel_constraint": kernel_constraint,
                "bias_constraint": bias_constraint,
                "name": f"dense_{base_name}",
            }
            return layers.Dense.from_config(**layer_config)

        elif self.layer == FeatureLayer.EMBEDDING and self.mode == FeatureMode.INT:
            layer_config = {
                "input_dim": self.cardinality,
                "output_dim": output_dim,
                "embeddings_initializer": embeddings_initializer,
                "embeddings_regularizer": embeddings_regularizer,
                "embeddings_constraint": embeddings_constraint,
                "mask_zero": mask_zero,
                "name": f"embedding_{base_name}",
            }
            return layers.Embedding.from_config(**layer_config)

        elif self.layer == FeatureLayer.IDENTITY:
            return layers.Identity(name=f"identity_{base_name}")

        else:
            raise FeatureArgumentError

    def to_module(
        self,
        output_dim: int,
        activation: nn.Module = None,
        bias: bool = True,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ) -> nn.Module:
        if self.layer == FeatureLayer.DENSE and self.mode in [
            FeatureMode.FLOAT,
            FeatureMode.ONE_HOT,
        ]:
            module_config = {
                "in_features": self.cardinality,
                "out_features": output_dim,
                "bias": bias,
                "device": device,
                "dtype": dtype,
            }
            if bias:
                return nn.Sequential(nn.Linear(**module_config), activation)
            else:
                return nn.Linear(**module_config)

        elif self.layer == FeatureLayer.EMBEDDING and self.mode == FeatureMode.INT:
            module_config = {
                "num_embeddings": self.cardinality,
                "embedding_dim": output_dim,
                "padding_idx": padding_idx,
                "max_norm": max_norm,
                "norm_type": norm_type,
                "scale_grad_by_freq": scale_grad_by_freq,
                "sparse": sparse,
                "device": device,
                "dtype": dtype,
            }
            return nn.Embedding(**module_config)

        elif self.layer == FeatureLayer.IDENTITY:
            return nn.Identity()

        else:
            raise FeatureArgumentError

    def to_dict(self):
        config_dict = asdict(self)
        config_dict.update({"format": self.format.value, "layer": self.layer.value})
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def feature_space_to_config(
    feature_space: keras.utils.FeatureSpace,
) -> Dict[str, FeatureConfig]:
    """Keras FeatureSpace to dictionary of FeatureConfig."""
    return NotImplemented
