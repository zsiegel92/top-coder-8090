import json
from pydantic import BaseModel

TRAINING_DATA_FILEPATH = "public_cases.json"
TEST_DATA_FILEPATH = "private_cases.json"


class Input(BaseModel):
    trip_duration_days: int
    miles_traveled: float
    total_receipts_amount: float


class InputOutputPair(BaseModel):
    input: Input
    expected_output: float


class InputArray(BaseModel):
    inputs: list[Input]


class InputOutputPairArray(BaseModel):
    input_output_pairs: list[InputOutputPair]


def load_training_data() -> list[InputOutputPair]:
    with open(TRAINING_DATA_FILEPATH, "r") as f:
        json_data = json.load(f)
    return InputOutputPairArray.model_validate(
        dict(input_output_pairs=json_data)
    ).input_output_pairs


def load_test_data() -> list[InputOutputPair]:
    with open(TEST_DATA_FILEPATH, "r") as f:
        json_data = json.load(f)
    return InputOutputPairArray.model_validate(
        dict(input_output_pairs=json_data)
    ).input_output_pairs