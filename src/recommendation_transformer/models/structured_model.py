from typing import Any, Callable, Dict, Iterable, Union

import keras
from keras import constraints
from keras import initializers
from keras import ops
from keras import regularizers
from keras.backend import KerasTensor

from recommendation_transformer import feature


class StructuredModel(keras.Model):
    def __init__(
        self,
        features: Dict[str, feature.FeatureConfig],
        embed_dim: int = 64,
        activation: Union[str, Callable] = "gelu",
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
        # ATTRIBUTES
        self.features = features
        self.embed_dim = embed_dim
        activation = keras.activations.get(activation)
        self.activation = activation

        # INITIALIZERS
        self._embeddings_initializer = initializers.get(embeddings_initializer)
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

        # REGULARIZERS
        self._embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)

        # CONSTRAINTS
        self._embeddings_constraint = constraints.get(embeddings_constraint)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        # FEATURE EMBEDDING LAYERS
        for feature_name in self.features:
            feature_layer_dict = {
                "output_dim": self.embed_dim,
                "activation": self.activation,
                "use_bias": True,
            }
            feature_layer_dict = {
                **feature_layer_dict,
                **self._dense_common_kwargs(),
                **self._embedding_common_kwargs(),
            }
            feature_layer = self.features[feature_name].to_layer(**feature_layer_dict)
            setattr(self, f"{feature_name}_embedding", feature_layer)
            print(
                f"""Feature layer set for feature `{feature_name}` with layer name 
                `{feature_layer.name}` and attribute name `{feature_name}_embedding`."""
            )
        print("Structured model instantiated from features.")

    def _dense_common_kwargs(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer

        return common_kwargs

    def _embedding_common_kwargs(self):
        common_kwargs = dict(
            embeddings_regularizer=self._embeddings_regularizer,
            activity_regularizer=self._activity_regularizer,
            embeddings_constraint=self._embeddings_constraint,
        )
        embeddings_initializer = self._embeddings_initializer.__class__.from_config(
            self._embeddings_initializer.get_config()
        )
        common_kwargs["embeddings_initializer"] = embeddings_initializer

        return common_kwargs

    def _embed(
        self,
        inputs: Dict[str, KerasTensor],
        feature_names: Iterable[str] = None,
        embed_mode: str = "stack",
    ) -> Union[KerasTensor, Dict[str, KerasTensor]]:
        if embed_mode not in ["dict", "stack"]:
            raise ValueError(
                f"""Invalid value for argument `embed_mode`.
                Expected one of [`dict`, `stack`], received: embed_mode={embed_mode}"""
            )

        if embed_mode == "stack":
            embed_features = ops.stack(
                [
                    getattr(self, f"{feature_name}_embedding")(inputs[feature_name])
                    for feature_name in feature_names
                ],
                axis=1,
            )
        elif embed_mode == "dict":
            embed_features = {
                feature_name: getattr(self, f"{feature_name}_embedding")(
                    inputs[feature_name]
                )
                for feature_name in feature_names
            }

        return embed_features

    # TO BE COMPLETED
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "features": {
                    feature_name: self.features[feature_name].to_dict()
                    for feature_name in self.features
                },
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        config["features"] = {
            feature_name: feature.FeatureConfig.from_dict(
                config["features"][feature_name]
            )
            for feature_name in config["features"]
        }
        return cls(**config)
