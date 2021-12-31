"""Features schema for structured/tabular models."""

from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel

import tensorflow as tf


class FeatureType(Enum):
    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"


class FieldType(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"


class FeatureSchema(BaseModel):
    name: str
    feature_type: FeatureType
    feature_dimension: int  # 1 if FeatureType.NUMERICAL`, N if FeatureType.CATEGORICAL`
    field_type: Optional[FieldType]
    description: Optional[str]
    minimum: Optional[Union[int, float]]
    maximum: Optional[Union[int, float]]


class InputFeaturesSchema(BaseModel):
    name: str
    ordered_features: List[FeatureSchema]
