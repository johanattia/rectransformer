"""Feature Reconstruction objective with TensorFlow"""

from typing import Callable, Dict, Union

import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2


class FeatureReconstructionLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        feature: Union[str, schema_pb2.Feature],
        loss_obj: tf.keras.losses.Loss,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = None,
    ):
        """_summary_

        Args:
            feature (Union[str, schema_pb2.Feature]): _description_
            loss_obj (tf.keras.losses.Loss): _description_
            reduction (str, optional): _description_. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): _description_. Defaults to None.
        """
        super().__init__(reduction=reduction, name=name)

        if isinstance(feature, schema_pb2.Feature):
            self.feature_name = feature.name
        elif isinstance(feature, str):
            self.feature_name = feature

        loss_obj.reduction = tf.keras.losses.Reduction.NONE
        self.loss_obj = loss_obj

    def call(
        self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """_summary_

        Args:
            y_true (Dict[str, tf.Tensor]): _description_
            y_pred (Dict[str, tf.Tensor]): _description_

        Returns:
            tf.Tensor: _description_
        """
        y_true = y_true[self.feature_name]
        y_pred = y_pred[self.feature_name]

        return self.loss_obj(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "feature": self.feature_name,
            "loss_obj": tf.keras.losses.serialize(self.loss_obj),
        }
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config["loss_obj"] = tf.keras.losses.get(config["loss_obj"])
        return cls(**config)
