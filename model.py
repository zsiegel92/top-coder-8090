import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from models_and_constants import load_training_data, Input, InputWithEngineeredFeatures, get_training_quantiles


def engineer_features(input_obj: Input) -> InputWithEngineeredFeatures:
    quantiles = get_training_quantiles()
    
    receipts_per_day = input_obj.total_receipts_amount / (input_obj.trip_duration_days + 1e-8)
    receipts_per_mile = input_obj.total_receipts_amount / (input_obj.miles_traveled + 1e-8)
    miles_per_day = input_obj.miles_traveled / (input_obj.trip_duration_days + 1e-8)
    
    log_receipts = float(np.log1p(input_obj.total_receipts_amount))
    log_miles = float(np.log1p(input_obj.miles_traveled))
    
    receipts_x_days = input_obj.total_receipts_amount * input_obj.trip_duration_days
    miles_x_days = input_obj.miles_traveled * input_obj.trip_duration_days
    
    high_receipt_flag = 1 if input_obj.total_receipts_amount > quantiles['receipts_q90'] else 0
    low_receipt_flag = 1 if input_obj.total_receipts_amount < quantiles['receipts_q10'] else 0
    long_trip_flag = 1 if input_obj.trip_duration_days > quantiles['days_q90'] else 0
    short_trip_flag = 1 if input_obj.trip_duration_days <= 1 else 0
    
    return InputWithEngineeredFeatures(
        trip_duration_days=input_obj.trip_duration_days,
        miles_traveled=input_obj.miles_traveled,
        total_receipts_amount=input_obj.total_receipts_amount,
        receipts_per_day=receipts_per_day,
        receipts_per_mile=receipts_per_mile,
        miles_per_day=miles_per_day,
        log_receipts=log_receipts,
        log_miles=log_miles,
        receipts_x_days=receipts_x_days,
        miles_x_days=miles_x_days,
        high_receipt_flag=high_receipt_flag,
        low_receipt_flag=low_receipt_flag,
        long_trip_flag=long_trip_flag,
        short_trip_flag=short_trip_flag
    )


def engineer_features_batch(inputs: list[Input], use_feature_engineering: bool = False) -> pd.DataFrame:
    if use_feature_engineering:
        enhanced_inputs = [engineer_features(inp) for inp in inputs]
        return pd.DataFrame([inp.model_dump() for inp in enhanced_inputs])
    else:
        return pd.DataFrame([inp.model_dump() for inp in inputs])


def train_model_with_features(hyperparams=None, use_feature_engineering=False):
    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_X = engineer_features_batch(inputs, use_feature_engineering)
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 0,
    }

    if hyperparams:
        default_params.update(hyperparams)

    if default_params.get("bootstrap", True):
        default_params["oob_score"] = True

    regressor = RandomForestRegressor(**default_params)
    regressor.fit(df_X.values, df_Y.values.ravel())
    return regressor


def train_production_model():
    best_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 4,
        "max_features": None,
        "bootstrap": True,
    }

    regressor = train_model_with_features(best_params, use_feature_engineering=True)
    joblib.dump(regressor, "production_model.pkl")
    return regressor


def load_model():
    import os
    if not os.path.exists("production_model.pkl"):
        print("No model found. Training default model...")
        return train_production_model()
    return joblib.load("production_model.pkl")
    
def predict(input: Input) -> float:
    regressor = load_model()
    
    if hasattr(regressor, "n_features_in_") and regressor.n_features_in_ > 3:
        enhanced_input = engineer_features(input)
        df_X = pd.DataFrame([enhanced_input.model_dump()])
    else:
        df_X = pd.DataFrame([input.model_dump()])

    Y = regressor.predict(df_X.values)
    return float(Y[0])


if __name__ == "__main__":
    print("Training production model...")
    train_production_model()
    print("Production model saved as production_model.pkl")
