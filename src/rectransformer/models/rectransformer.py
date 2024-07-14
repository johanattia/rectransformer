from typing import Any, Callable, Dict, Iterable, Tuple, Type, Union

import keras
from keras import layers
from keras.backend import KerasTensor

from rectransformer import feature
from rectransformer.layers import (
    AttentionPooling,
    Predictor,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from rectransformer.models import StructuredModel


class FeatureConsistencyError(Exception):
    pass


class RecTransformer(StructuredModel):
    def __init__(
        self,
        output_dim: int,
        features: Dict[str, feature.FeatureConfig],
        encoder_features: Iterable[str],
        decoder_features: Iterable[str],
        predictor_features: Iterable[str],
        num_heads: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
        embed_dim: int = 64,
        activation: Union[str, Callable] = "gelu",
        embed_mode: str = "concat",
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
        self._check_init_feature_consistency(
            features=features,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            predictor_features=predictor_features,
        )
        super().__init__(
            features=features,
            embed_dim=embed_dim,
            activation=activation,
            embed_mode=embed_mode,
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
        # FEATURES
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.predictor_features = predictor_features

        # ARCHITECTURE ATTRIBUTES
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks

        # ARCHITECTURE LAYERS

    def _check_init_feature_consistency(
        features: Dict[str, feature.FeatureConfig],
        encoder_features: Iterable[str],
        decoder_features: Iterable[str],
        predictor_features: Iterable[str],
    ):
        """Check consistency of provided features for instantiation."""
        features_keys = set(features.keys())
        block_features = set(encoder_features + decoder_features + predictor_features)
        feature_difference = features_keys ^ block_features

        if feature_difference:
            raise FeatureConsistencyError(
                f"""Following features are not consistent between `features`, 
                `encoder_features` and `decoder_features` args: {feature_difference}."""
            )

    def _check_call_feature_consistency(self, inputs: Dict[str, KerasTensor]):
        """Check consistency of provided features for instantiation."""
        features_keys = set(self.features.keys())
        inputs_key = set(inputs.keys())
        feature_difference = features_keys ^ inputs_key

        if feature_difference:
            raise FeatureConsistencyError(
                f"""Following features are not consistent between self `features`, 
                call `inputs` arg: {feature_difference}."""
            )

    def _encoder(self, encoder_inputs: KerasTensor) -> KerasTensor:
        return encoder_inputs

    def _decoder(self, decoder_inputs, encoder_output: KerasTensor) -> KerasTensor:
        return decoder_inputs

    def call(self, inputs: Dict[str, KerasTensor], training: bool = False):
        self._check_call_feature_consistency(inputs)

        # Inputs Embeddings
        encoder_embedding = self._embed(inputs, self.encoder_features)
        decoder_embedding = self._embed(inputs, self.decoder_features)

        # Transformer Encoder-Decoder
        encoder_output = self._encoder(encoder_embedding)
        decoder_output = self._decoder(decoder_embedding, encoder_output)

        return NotImplemented

    def get_config(self) -> Dict[str, Any]:
        # return super().get_config()
        return NotImplemented
