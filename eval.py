import pandas as pd
from models_and_constants import load_training_data
from model import load_model, engineer_features


def batch_predict(model, inputs):
    """Make batch predictions efficiently"""
    # Check if model uses feature engineering
    use_feature_engineering = (
        hasattr(model, "n_features_in_") and model.n_features_in_ > 3
    )

    if use_feature_engineering:
        # Apply feature engineering to all inputs
        enhanced_inputs = [engineer_features(inp) for inp in inputs]
        df_X = pd.DataFrame([inp.model_dump() for inp in enhanced_inputs])
    else:
        # Use raw inputs
        df_X = pd.DataFrame([inp.model_dump() for inp in inputs])

    # Make batch predictions
    predictions = model.predict(df_X.values)
    return predictions


def evaluate_public_cases():
    """Evaluate on public test cases and save results"""
    print("Loading model...")
    model = load_model()

    print("Loading public test cases...")
    public_data = load_training_data()  # Public cases are actually the training data

    inputs = [point.input for point in public_data]
    true_outputs = [point.expected_output for point in public_data]

    print(f"Making predictions for {len(inputs)} public cases...")
    predictions = batch_predict(model, inputs)

    # Create results dataframe
    results = []
    for i, (input_obj, true_output, prediction) in enumerate(
        zip(inputs, true_outputs, predictions)
    ):
        results.append(
            {
                "case_id": i,
                "trip_duration_days": input_obj.trip_duration_days,
                "miles_traveled": input_obj.miles_traveled,
                "total_receipts_amount": input_obj.total_receipts_amount,
                "true_output": true_output,
                "predicted_output": float(prediction),
                "absolute_error": abs(true_output - prediction),
                "relative_error_pct": (
                    100 * abs(true_output - prediction) / true_output
                    if true_output != 0
                    else 0
                ),
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv("public_output.csv", index=False)

    print(f"Public results saved to public_output.csv")

    # Print summary statistics
    mean_error = df_results["absolute_error"].mean()
    max_error = df_results["absolute_error"].max()
    close_matches_1 = (df_results["absolute_error"] <= 1.0).sum()
    exact_matches = (df_results["absolute_error"] <= 0.01).sum()

    print(f"\nPublic Results Summary:")
    print(f"  Total cases: {len(results)}")
    print(
        f"  Exact matches (±$0.01): {exact_matches} ({100*exact_matches/len(results):.1f}%)"
    )
    print(
        f"  Close matches (±$1.00): {close_matches_1} ({100*close_matches_1/len(results):.1f}%)"
    )
    print(f"  Average error: ${mean_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")

    return model


def evaluate_private_cases(model):
    """Evaluate on private test cases and save results"""
    try:
        print("Loading private test cases...")
        import json
        from models_and_constants import Input

        # Load private cases directly as they only contain inputs
        with open(
            "/Users/zach/Dropbox/code/top-coder-challenge/private_cases.json", "r"
        ) as f:
            private_inputs_json = json.load(f)

        # Convert to Input objects
        inputs = [Input(**case) for case in private_inputs_json]

        print(f"Making predictions for {len(inputs)} private cases...")
        predictions = batch_predict(model, inputs)

        # Create results dataframe (predictions only since no ground truth)
        results = []
        for i, (input_obj, prediction) in enumerate(zip(inputs, predictions)):
            results.append(
                {
                    "case_id": i,
                    "trip_duration_days": input_obj.trip_duration_days,
                    "miles_traveled": input_obj.miles_traveled,
                    "total_receipts_amount": input_obj.total_receipts_amount,
                    "predicted_output": float(prediction),
                }
            )

        df_results = pd.DataFrame(results)
        df_results.to_csv("private_output.csv", index=False)

        print(f"Private results saved to private_output.csv")
        print(f"Total private cases: {len(results)}")
        print("Note: No ground truth available for private cases")

    except FileNotFoundError:
        print("Private test cases not found")
        print("Note: No private test data available")


def main():
    print("=" * 60)
    print("MODEL EVALUATION - BATCH PREDICTION")
    print("=" * 60)

    # Load model once and use for both evaluations
    model = evaluate_public_cases()
    evaluate_private_cases(model)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - public_output.csv (with ground truth)")
    print("  - private_output.csv (with ground truth if available)")


if __name__ == "__main__":
    main()
