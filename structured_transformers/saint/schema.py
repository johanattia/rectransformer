"""Features schema for structured/tabular models."""

from typing import Dict, Optional, Union
from pydantic import BaseModel, validator

import tensorflow as tf


class FeatureSchema(BaseModel):
    name: str
    feature_type: str
    feature_dimension: int  # 1 if `NUMERICAL`, N if `CATEGORICAL`
    field_type: Optional[Union[tf.io.FixedLenFeature, tf.dtypes.DType]] = None
    description: Optional[str] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None

    @validator("feature_type")
    def feature_type_match(cls, value):
        if not (
            isinstance(value, str)
            and (value.upper() not in ["NUMERICAL", "CATEGORICAL"])
        ):
            raise ValueError("`feature_type` must be in [`NUMERICAL`, `CATEGORICAL`]")
        return value.upper()


class FeaturesMapping(BaseModel):
    name: str
    mapping: Dict[str, FeatureSchema]
