import json
from pydantic import BaseModel

TRAINING_DATA_FILEPATH = "public_cases.json"
TEST_DATA_FILEPATH = "private_cases.json"


class Input(BaseModel):
    trip_duration_days: int
    miles_traveled: float
    total_receipts_amount: float


class InputWithEngineeredFeatures(BaseModel):
    trip_duration_days: int
    miles_traveled: float
    total_receipts_amount: float
    receipts_per_day: float
    receipts_per_mile: float
    miles_per_day: float
    log_receipts: float
    log_miles: float
    receipts_x_days: float
    miles_x_days: float
    high_receipt_flag: int
    low_receipt_flag: int
    long_trip_flag: int
    short_trip_flag: int


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


def get_training_quantiles():
    training_data = load_training_data()
    receipts = [point.input.total_receipts_amount for point in training_data]
    days = [point.input.trip_duration_days for point in training_data]
    
    import numpy as np
    return {
        'receipts_q90': float(np.quantile(receipts, 0.9)),
        'receipts_q10': float(np.quantile(receipts, 0.1)),
        'days_q90': float(np.quantile(days, 0.9))
    }