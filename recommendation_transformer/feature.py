from dataclasses import asdict
from enum import Enum
from typing import Callable, Union

from keras import layers
from pydantic.dataclasses import dataclass


class FeatureArgumentError(Exception):
  pass


class FeatureMode(Enum):
  FLOAT = "float"
  INT = "int"
  ONE_HOT = "one_hot"


class FeatureLayer(Enum):
  DENSE = "dense"
  EMBEDDING = "embedding"


@dataclass
class FeatureConfig:
  """Dataclass to store needed information about features."""

  name: str
  mode: FeatureMode
  cardinality: int
  layer: FeatureLayer

  def to_layer(
      self, 
      output_dim: int = 32,
      activation: Union[str, Callable] = None,
      use_bias: bool =True,
      embeddings_initializer:Union[str, Callable]="uniform",
      kernel_initializer:Union[str, Callable]="glorot_uniform",
      bias_initializer:Union[str, Callable]="zeros",
      embeddings_regularizer:Union[str, Callable]=None,
      kernel_regularizer:Union[str, Callable]=None,
      bias_regularizer:Union[str, Callable]=None,
      activity_regularizer:Union[str, Callable]=None,
      embeddings_constraint:Union[str, Callable]=None,
      kernel_constraint:Union[str, Callable]=None,
      bias_constraint:Union[str, Callable]=None,
      mask_zero=False,
      suffix: str = "", 
      prefix: str = "",
      **kwargs
  ) -> layers.Layer:
    name = "_".join([prefix, self.name, suffix])

    if self.layer == FeatureLayer.DENSE and self.mode in [FeatureMode.FLOAT, FeatureMode.ONE_HOT]:
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
        "name": name,
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
          "name": name,
      }
      return layers.Embedding.from_config(**layer_config)
    
    else:
      raise FeatureArgumentError

  def to_dict(self):
    config_dict = asdict(self)
    config_dict["format"] = self.format.value
    config_dict["layer"] = self.layer.value

    return config_dict
  
  @classmethod
  def from_dict(cls, config_dict):
    return cls(**config_dict)
