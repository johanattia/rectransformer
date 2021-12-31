"""Features schema for structured/tabular models."""

import json

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel


class FeatureType(Enum):
    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"

    @classmethod
    def from_string(cls, value: str):
        if value.upper() == "CATEGORICAL":
            return cls.CATEGORICAL
        elif value.upper() == "NUMERICAL":
            return cls.NUMERICAL
        else:
            raise ValueError("`value` must be in [`CATEGORICAL`, `NUMERICAL`]")


class FieldType(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"

    @classmethod
    def from_string(cls, value: str):
        if value.upper() == "INT":
            return cls.INT
        elif value.upper() == "FLOAT":
            return cls.FLOAT
        elif value.upper() == "STRING":
            return cls.STRING
        else:
            raise ValueError("`value` must be in [`INT`, `FLOAT`, `STRING`]")


class FeatureSchema(BaseModel):
    name: str
    feature_type: FeatureType
    feature_dimension: int  # 1 if FeatureType.NUMERICAL, N if FeatureType.CATEGORICAL
    field_type: Optional[FieldType]
    description: Optional[str]
    minimum: Optional[Union[int, float]]
    maximum: Optional[Union[int, float]]


class InputFeaturesSchema(BaseModel):
    name: str
    ordered_features: List[FeatureSchema]


def feature_from_json(json_string: str) -> FeatureSchema:
    """[summary]

    Args:
        json_string (str): [description]

    Returns:
        FeatureSchema: [description]
    """
    feature_config = json.loads(json_string)

    feature_config["feature_type"] = FeatureType.from_string(
        feature_config["feature_type"]
    )
    feature_config["field_type"] = FieldType.from_string(feature_config["field_type"])

    return FeatureSchema(**feature_config)


def input_schema_from_json(json_string: str) -> InputFeaturesSchema:
    """[summary]

    Args:
        json_string (str): [description]

    Returns:
        InputFeaturesSchema: [description]
    """
    input_config = json.loads(json_string)

    for i, feature in enumerate(input_config["ordered_features"]):
        feature["feature_type"] = FeatureType.from_string(feature["feature_type"])
        feature["field_type"] = FieldType.from_string(feature["field_type"])

        input_config["ordered_features"][i] = FeatureSchema(**feature)

    return InputFeaturesSchema(**input_config)
