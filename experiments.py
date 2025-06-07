import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import seaborn as sns
from sklearn.manifold import TSNE
from itertools import product
from typing import Dict, List, Any
from models_and_constants import load_training_data
from model import engineer_features_batch, train_model_with_features, load_model


def cross_validate_model(
    hyperparams: Dict[str, Any], k_folds: int = 5, use_feature_engineering: bool = False
) -> Dict[str, Any]:
    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_X = engineer_features_batch(inputs, use_feature_engineering)
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    X, y = df_X.values, df_Y.values.ravel()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_scores = []
    fold_mse = []
    fold_r2 = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_mse.append(mse)
        fold_r2.append(r2)
        fold_scores.append(r2)

    return {
        "mean_cv_score": np.mean(fold_scores),
        "std_cv_score": np.std(fold_scores),
        "mean_mse": np.mean(fold_mse),
        "std_mse": np.std(fold_mse),
        "mean_r2": np.mean(fold_r2),
        "std_r2": np.std(fold_r2),
        "hyperparams": hyperparams,
    }


def generate_hyperparameter_grid(
    custom_ranges: Dict[str, List] | None = None,
) -> List[Dict[str, Any]]:
    if custom_ranges:
        param_grid = custom_ranges
    else:
        param_grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def expand_parameter_ranges(
    best_params: Dict[str, Any], current_ranges: Dict[str, List]
) -> Dict[str, List]:
    expanded_ranges = current_ranges.copy()

    # n_estimators: expand if at boundary
    if best_params["n_estimators"] == max(current_ranges["n_estimators"]):
        max_val = max(current_ranges["n_estimators"])
        new_values = [max_val * 2, max_val * 4]
        expanded_ranges["n_estimators"] = current_ranges["n_estimators"] + new_values
        print(f"Expanding n_estimators: adding {new_values}")
    elif best_params["n_estimators"] == min(current_ranges["n_estimators"]):
        min_val = min(current_ranges["n_estimators"])
        new_values = [max(1, min_val // 2), max(1, min_val // 4)]
        expanded_ranges["n_estimators"] = new_values + current_ranges["n_estimators"]
        print(f"Expanding n_estimators: adding {new_values}")

    # max_depth: expand if at boundary (skip None)
    numeric_depths = [d for d in current_ranges["max_depth"] if d is not None]
    if numeric_depths and best_params["max_depth"] == max(numeric_depths):
        max_val = max(numeric_depths)
        new_values = [max_val + 20, max_val + 50]
        expanded_ranges["max_depth"] = current_ranges["max_depth"] + new_values
        print(f"Expanding max_depth: adding {new_values}")
    elif numeric_depths and best_params["max_depth"] == min(numeric_depths):
        min_val = min(numeric_depths)
        new_values = [max(1, min_val - 5), max(1, min_val - 10)]
        expanded_ranges["max_depth"] = new_values + current_ranges["max_depth"]
        print(f"Expanding max_depth: adding {new_values}")

    # min_samples_split: expand if at boundary
    if best_params["min_samples_split"] == max(current_ranges["min_samples_split"]):
        max_val = max(current_ranges["min_samples_split"])
        new_values = [max_val + 5, max_val + 10, max_val + 20]
        expanded_ranges["min_samples_split"] = (
            current_ranges["min_samples_split"] + new_values
        )
        print(f"Expanding min_samples_split: adding {new_values}")

    # min_samples_leaf: expand if at boundary
    if best_params["min_samples_leaf"] == max(current_ranges["min_samples_leaf"]):
        max_val = max(current_ranges["min_samples_leaf"])
        new_values = [max_val + 2, max_val + 5, max_val + 10]
        expanded_ranges["min_samples_leaf"] = (
            current_ranges["min_samples_leaf"] + new_values
        )
        print(f"Expanding min_samples_leaf: adding {new_values}")

    return expanded_ranges


def fast_hyperparameter_search(
    custom_ranges: Dict[str, List] | None = None,
    max_experiments: int = 20,
) -> pd.DataFrame:
    print("Running fast hyperparameter search...")

    if custom_ranges is None:
        current_ranges = {
            "n_estimators": [100, 500, 1000],
            "max_depth": [None, 30],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", None],
            "bootstrap": [True],
        }
    else:
        current_ranges = custom_ranges

    print("Parameter ranges:")
    for param, values in current_ranges.items():
        print(f"  {param}: {values}")

    grid = generate_hyperparameter_grid(current_ranges)

    if len(grid) > max_experiments:
        print(f"Grid has {len(grid)} combinations, sampling {max_experiments}")
        np.random.seed(42)
        indices = np.random.choice(len(grid), max_experiments, replace=False)
        grid = [grid[i] for i in indices]

    print(f"Testing {len(grid)} parameter combinations...")

    results = []
    for i, params in enumerate(grid):
        if i % 5 == 0:
            print(f"Progress: {i+1}/{len(grid)}")
        try:
            cv_results = cross_validate_model(
                params, 2, use_feature_engineering=True
            )  # 2-fold for max speed
            results.append(cv_results)
        except Exception as e:
            print(f"Failed for params {params}: {e}")
            continue

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mean_cv_score", ascending=False)

    print(f"\nTop 5 results:")
    for i, (_, row) in enumerate(df_results.head(5).iterrows()):
        params = row["hyperparams"]
        score = row["mean_cv_score"]
        print(f"  {i+1}. R²={score:.6f}: {params}")

    # Check if best params hit boundaries
    best_params = df_results.iloc[0]["hyperparams"]
    print(f"\nBoundary analysis for best params: {best_params}")

    boundary_hits = []
    if best_params["n_estimators"] == max(current_ranges["n_estimators"]):
        boundary_hits.append(f"n_estimators hit max ({best_params['n_estimators']})")
    if best_params["n_estimators"] == min(current_ranges["n_estimators"]):
        boundary_hits.append(f"n_estimators hit min ({best_params['n_estimators']})")

    numeric_depths = [d for d in current_ranges["max_depth"] if d is not None]
    if numeric_depths and best_params["max_depth"] == max(numeric_depths):
        boundary_hits.append(f"max_depth hit max ({best_params['max_depth']})")

    if best_params["min_samples_split"] == max(current_ranges["min_samples_split"]):
        boundary_hits.append(
            f"min_samples_split hit max ({best_params['min_samples_split']})"
        )
    if best_params["min_samples_split"] == min(current_ranges["min_samples_split"]):
        boundary_hits.append(
            f"min_samples_split hit min ({best_params['min_samples_split']})"
        )

    if best_params["min_samples_leaf"] == max(current_ranges["min_samples_leaf"]):
        boundary_hits.append(
            f"min_samples_leaf hit max ({best_params['min_samples_leaf']})"
        )
    if best_params["min_samples_leaf"] == min(current_ranges["min_samples_leaf"]):
        boundary_hits.append(
            f"min_samples_leaf hit min ({best_params['min_samples_leaf']})"
        )

    if boundary_hits:
        print("*** BOUNDARY HITS DETECTED ***")
        for hit in boundary_hits:
            print(f"  {hit}")
        print("Consider expanding ranges and re-running!")

        # Suggest expanded ranges
        print("\nSuggested expanded ranges:")
        new_ranges = current_ranges.copy()

        if best_params["n_estimators"] == max(current_ranges["n_estimators"]):
            max_val = max(current_ranges["n_estimators"])
            new_ranges["n_estimators"] = current_ranges["n_estimators"] + [
                max_val * 2,
                max_val * 3,
            ]

        if numeric_depths and best_params["max_depth"] == max(numeric_depths):
            max_val = max(numeric_depths)
            new_ranges["max_depth"] = current_ranges["max_depth"] + [
                max_val + 30,
                max_val + 60,
            ]

        if best_params["min_samples_split"] == max(current_ranges["min_samples_split"]):
            max_val = max(current_ranges["min_samples_split"])
            new_ranges["min_samples_split"] = current_ranges["min_samples_split"] + [
                max_val + 10,
                max_val + 20,
            ]

        if best_params["min_samples_leaf"] == max(current_ranges["min_samples_leaf"]):
            max_val = max(current_ranges["min_samples_leaf"])
            new_ranges["min_samples_leaf"] = current_ranges["min_samples_leaf"] + [
                max_val + 5,
                max_val + 10,
            ]

        for param, values in new_ranges.items():
            if values != current_ranges[param]:
                print(f"  {param}: {values}")
    else:
        print("No boundary hits detected. Current ranges seem adequate.")

    return df_results


def run_adaptive_hyperparameter_experiment() -> None:
    print("Running fast adaptive hyperparameter search...")
    results_df = fast_hyperparameter_search(max_experiments=50)

    # Save detailed results
    results_df.to_csv("adaptive_hyperparameter_results.csv", index=False)
    print("Detailed results saved to adaptive_hyperparameter_results.csv")

    # Print top 10 results
    print("\nTop 10 hyperparameter combinations:")
    print("=" * 100)
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        print(f"\nRank {i+1}:")
        print(f"  Mean CV R²: {row['mean_cv_score']:.6f} ± {row['std_cv_score']:.6f}")
        print(f"  Mean MSE: {row['mean_mse']:.2f} ± {row['std_mse']:.2f}")
        print(f"  Hyperparams: {row['hyperparams']}")

    # Train final model with absolute best hyperparameters
    best_params = results_df.iloc[0]["hyperparams"]
    best_score = results_df.iloc[0]["mean_cv_score"]

    print(f"\n{'='*80}")
    print(f"TRAINING FINAL PRODUCTION MODEL")
    print(f"{'='*80}")
    print(f"Best CV R²: {best_score:.6f}")
    print(f"Best hyperparameters: {best_params}")

    final_model = train_model_with_features(best_params, use_feature_engineering=True)

    # Save as production model
    joblib.dump(final_model, "production_model.pkl")
    print(f"\n🎯 PRODUCTION MODEL SAVED as production_model.pkl")
    print(f"Model trained with feature engineering and best hyperparameters")

    # Test the final model on training data to verify
    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_X = engineer_features_batch(inputs, use_feature_engineering=True)
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    Y_hat = final_model.predict(df_X.values)
    from sklearn.metrics import mean_squared_error, r2_score

    final_r2 = r2_score(df_Y.values, Y_hat)
    final_mse = mean_squared_error(df_Y.values, Y_hat)

    print(f"\nFinal model training performance:")
    print(f"  R² on full training set: {final_r2:.6f}")
    print(f"  MSE on full training set: {final_mse:.2f}")
    print(f"  Expected CV R²: {best_score:.6f}")


def test_expanded_ranges() -> None:
    print("Testing with expanded ranges based on previous boundary hits...")

    # Focused on most promising combinations
    expanded_ranges = {
        "n_estimators": [1000, 2000],
        "max_depth": [30, 60, None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt", None],
        "bootstrap": [True],
    }

    results_df = fast_hyperparameter_search(expanded_ranges, max_experiments=12)

    # Save results and train best model
    results_df.to_csv("expanded_range_results.csv", index=False)
    print("Results saved to expanded_range_results.csv")

    best_params = results_df.iloc[0]["hyperparams"]
    best_score = results_df.iloc[0]["mean_cv_score"]

    print(f"\nTraining final model with expanded range best params...")
    print(f"Best R²: {best_score:.6f}")
    print(f"Best params: {best_params}")

    final_model = train_model_with_features(best_params, use_feature_engineering=True)
    joblib.dump(final_model, "production_model.pkl")
    print("🎯 PRODUCTION MODEL SAVED as production_model.pkl")


def hyperparameter_search(
    k_folds: int = 5, max_experiments: int = 50, quick_search: bool = False
) -> pd.DataFrame:
    if quick_search:
        custom_ranges = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True],
        }
        grid = generate_hyperparameter_grid(custom_ranges)
    else:
        grid = generate_hyperparameter_grid()

    if len(grid) > max_experiments:
        print(f"Grid has {len(grid)} combinations, sampling {max_experiments}")
        np.random.seed(42)
        indices = np.random.choice(len(grid), max_experiments, replace=False)
        grid = [grid[i] for i in indices]

    results = []

    for i, params in enumerate(grid):
        print(f"Experiment {i+1}/{len(grid)}: {params}")
        try:
            cv_results = cross_validate_model(params, k_folds)
            results.append(cv_results)
        except Exception as e:
            print(f"Failed for params {params}: {e}")
            continue

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mean_cv_score", ascending=False)

    return df_results


def run_hyperparameter_experiment() -> None:
    print("Starting hyperparameter search...")
    results_df = hyperparameter_search(k_folds=5, max_experiments=50)

    results_df.to_csv("hyperparameter_results.csv", index=False)
    print("Results saved to hyperparameter_results.csv")

    print("\nTop 5 hyperparameter combinations:")
    print("=" * 80)
    for i, (_, row) in enumerate(results_df.head().iterrows()):
        print(f"\nRank {i+1}:")
        print(f"  Mean CV R²: {row['mean_cv_score']:.4f} ± {row['std_cv_score']:.4f}")
        print(f"  Mean MSE: {row['mean_mse']:.2f} ± {row['std_mse']:.2f}")
        print(f"  Hyperparams: {row['hyperparams']}")

    best_params = results_df.iloc[0]["hyperparams"]
    print(f"\nTraining final model with best hyperparameters: {best_params}")
    best_model = train_model_with_features(best_params)
    joblib.dump(best_model, "best_trained_model.pkl")
    print("Best model saved as best_trained_model.pkl")


def inspect_model():
    regressor = load_model()
    tree_to_plot = regressor.estimators_[0]
    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    use_feature_engineering = (
        hasattr(regressor, "n_features_in_") and regressor.n_features_in_ > 3
    )
    df_X = engineer_features_batch(inputs, use_feature_engineering)

    Y_hat = regressor.predict(df_X.values)
    oob_score = regressor.oob_score_
    r2 = r2_score(df_Y.values, Y_hat)
    MSE = mean_squared_error(df_Y.values, Y_hat)
    print(f"oob_score: {oob_score}")
    print(f"r2: {r2}")
    print(f"MSE: {MSE}")

    importances = regressor.feature_importances_
    feature_importance_dict = dict(zip(df_X.columns, importances))
    print("Feature Importances:")
    for feature, importance in sorted(
        feature_importance_dict.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feature}: {importance:.4f}")

    plt.figure(figsize=(25, 15))
    plot_tree(
        tree_to_plot,
        feature_names=df_X.columns.tolist(),
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4,
    )
    plt.title("Decision Tree from Random Forest (Limited Depth)")
    plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df_Y.values.ravel(), y=Y_hat, alpha=0.6)
    plt.plot(
        [df_Y.values.min(), df_Y.values.max()],
        [df_Y.values.min(), df_Y.values.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("True Values (Y)")
    plt.ylabel("Predicted Values (Y-hat)")
    plt.title("Y vs Y-hat")
    plt.savefig("y_vs_yhat.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Computing TSNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(df_X.values)

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    scatter1 = axes[0].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=df_Y.values.ravel(), cmap="viridis", alpha=0.7
    )
    axes[0].set_title("TSNE: True Y Values")
    axes[0].set_xlabel("TSNE 1")
    axes[0].set_ylabel("TSNE 2")
    plt.colorbar(scatter1, ax=axes[0])

    scatter2 = axes[1].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=Y_hat, cmap="viridis", alpha=0.7
    )
    axes[1].set_title("TSNE: Predicted Y-hat Values")
    axes[1].set_xlabel("TSNE 1")
    axes[1].set_ylabel("TSNE 2")
    plt.colorbar(scatter2, ax=axes[1])

    diff = df_Y.values.ravel() - Y_hat
    scatter3 = axes[2].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=diff, cmap="RdBu", alpha=0.7
    )
    axes[2].set_title("TSNE: Prediction Error (Y - Y-hat)")
    axes[2].set_xlabel("TSNE 1")
    axes[2].set_ylabel("TSNE 2")
    plt.colorbar(scatter3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("tsne_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_enhanced_model() -> None:
    print("Testing enhanced model with feature engineering...")

    print("Training baseline model...")
    baseline_model = train_model_with_features(use_feature_engineering=False)

    print("Training enhanced model...")
    enhanced_model = train_model_with_features(use_feature_engineering=True)

    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    df_X_baseline = engineer_features_batch(inputs, use_feature_engineering=False)
    baseline_pred = baseline_model.predict(df_X_baseline.values)
    baseline_errors = np.abs(df_Y.values.ravel() - baseline_pred)

    df_X_enhanced = engineer_features_batch(inputs, use_feature_engineering=True)
    enhanced_pred = enhanced_model.predict(df_X_enhanced.values)
    enhanced_errors = np.abs(df_Y.values.ravel() - enhanced_pred)

    worst_indices = np.argsort(baseline_errors)[-20:][::-1]

    print(f"\nEdge Case Performance Comparison:")
    print(
        f"{'Rank':<5} {'True':<8} {'Baseline':<10} {'Enhanced':<10} {'Improvement':<12}"
    )
    print("-" * 50)

    improvements = []
    for i, idx in enumerate(worst_indices):
        true_val = df_Y.iloc[idx].values[0]
        baseline_err = baseline_errors[idx]
        enhanced_err = enhanced_errors[idx]
        improvement = baseline_err - enhanced_err
        improvement_pct = 100 * improvement / baseline_err
        improvements.append(improvement)

        print(
            f"{i+1:<5} {true_val:<8.1f} {baseline_err:<10.1f} {enhanced_err:<10.1f} {improvement_pct:<12.1f}%"
        )

    print("\nOverall Edge Case Results:")
    print(f"  Baseline mean error: {np.mean(baseline_errors[worst_indices]):.1f}")
    print(f"  Enhanced mean error: {np.mean(enhanced_errors[worst_indices]):.1f}")
    print(f"  Mean improvement: {np.mean(improvements):.1f}")
    print(f"  Cases improved: {sum(1 for x in improvements if x > 0)}/20")

    print("\nFeature Importance (Enhanced Model):")
    feature_names = df_X_enhanced.columns.tolist()
    importances = enhanced_model.feature_importances_
    sorted_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    for name, importance in sorted_features[:10]:
        print(f"  {name}: {importance:.4f}")


def find_edge_cases(num_cases: int = 20) -> None:
    print(f"Finding {num_cases} worst-performing examples...")

    regressor = load_model()

    training_data = load_training_data()
    inputs = [point.input for point in training_data]
    df_X = pd.DataFrame([point.input.model_dump() for point in training_data])
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    use_feature_engineering = (
        hasattr(regressor, "n_features_in_") and regressor.n_features_in_ > 3
    )
    df_X_pred = engineer_features_batch(inputs, use_feature_engineering)

    Y_hat = regressor.predict(df_X_pred.values)

    errors = np.abs(df_Y.values.ravel() - Y_hat)

    worst_indices = np.argsort(errors)[-num_cases:][::-1]

    edge_cases = []
    for i, idx in enumerate(worst_indices):
        case = {
            "rank": i + 1,
            "index": int(idx),
            "true_value": float(df_Y.iloc[idx].values[0]),
            "predicted_value": float(Y_hat[idx]),
            "absolute_error": float(errors[idx]),
            "relative_error_pct": float(100 * errors[idx] / df_Y.iloc[idx].values[0]),
            "total_receipts_amount": float(df_X.iloc[idx]["total_receipts_amount"]),
            "trip_duration_days": float(df_X.iloc[idx]["trip_duration_days"]),
            "miles_traveled": float(df_X.iloc[idx]["miles_traveled"]),
        }
        edge_cases.append(case)

        print(
            f"Rank {i+1}: True={case['true_value']:.1f}, Pred={case['predicted_value']:.1f}, "
            f"Error={case['absolute_error']:.1f} ({case['relative_error_pct']:.1f}%)"
        )

    edge_cases_df = pd.DataFrame(edge_cases)
    edge_cases_df.to_csv("edge_cases.csv", index=False)
    print(f"\nEdge cases saved to edge_cases.csv")

    print(f"\nEdge Case Summary:")
    print(f"  Mean absolute error: {edge_cases_df['absolute_error'].mean():.1f}")
    print(f"  Max absolute error: {edge_cases_df['absolute_error'].max():.1f}")
    print(f"  Mean relative error: {edge_cases_df['relative_error_pct'].mean():.1f}%")
    print(f"  Max relative error: {edge_cases_df['relative_error_pct'].max():.1f}%")


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage:")
        print(
            "  python experiments.py fast-search            # RECOMMENDED: Fast boundary-aware search"
        )
        print(
            "  python experiments.py expanded-search        # Test with expanded ranges"
        )
        print(
            "  python experiments.py adaptive-search        # Full adaptive search (slower)"
        )
        print(
            "  python experiments.py hyperparameter-search  # Standard hyperparameter search"
        )
        print("  python experiments.py quick-search")
        print("  python experiments.py test-cv")
        print("  python experiments.py edge-cases [N]")
        print("  python experiments.py test-enhanced")
        print("  python experiments.py compare-cv")
        print("  python experiments.py inspect")
    elif args[0] == "fast-search":
        results = fast_hyperparameter_search(max_experiments=20)
        results.to_csv("fast_search_results.csv", index=False)
        print("Results saved to fast_search_results.csv")
    elif args[0] == "expanded-search":
        test_expanded_ranges()
    elif args[0] == "adaptive-search":
        run_adaptive_hyperparameter_experiment()
    elif args[0] == "hyperparameter-search":
        run_hyperparameter_experiment()
    elif args[0] == "quick-search":
        print("Running quick hyperparameter search...")
        results_df = hyperparameter_search(
            k_folds=3, max_experiments=12, quick_search=True
        )
        print(f"\nCompleted {len(results_df)} experiments")
        print("\nTop 3 results:")
        print(results_df.head(3)[["mean_cv_score", "mean_mse", "hyperparams"]])
    elif args[0] == "test-cv":
        print("Testing cross-validation with default parameters...")
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
        }
        cv_results = cross_validate_model(default_params, k_folds=5)
        print(f"CV Results: {cv_results}")
    elif args[0] == "edge-cases":
        num_cases = int(args[1]) if len(args) > 1 else 20
        find_edge_cases(num_cases)
    elif args[0] == "test-enhanced":
        test_enhanced_model()
    elif args[0] == "compare-cv":
        print("Comparing cross-validation performance...")
        best_params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 4,
            "max_features": None,
            "bootstrap": True,
        }

        print("Testing baseline model...")
        baseline_cv = cross_validate_model(
            best_params, k_folds=5, use_feature_engineering=False
        )

        print("Testing enhanced model...")
        enhanced_cv = cross_validate_model(
            best_params, k_folds=5, use_feature_engineering=True
        )

        print(f"\nCross-Validation Results:")
        print(
            f"  Baseline R²: {baseline_cv['mean_cv_score']:.4f} ± {baseline_cv['std_cv_score']:.4f}"
        )
        print(
            f"  Enhanced R²: {enhanced_cv['mean_cv_score']:.4f} ± {enhanced_cv['std_cv_score']:.4f}"
        )
        print(
            f"  Improvement: {enhanced_cv['mean_cv_score'] - baseline_cv['mean_cv_score']:.4f}"
        )

        print(
            f"\n  Baseline MSE: {baseline_cv['mean_mse']:.1f} ± {baseline_cv['std_mse']:.1f}"
        )
        print(
            f"  Enhanced MSE: {enhanced_cv['mean_mse']:.1f} ± {enhanced_cv['std_mse']:.1f}"
        )
        print(
            f"  MSE Reduction: {baseline_cv['mean_mse'] - enhanced_cv['mean_mse']:.1f}"
        )
    elif args[0] == "inspect":
        inspect_model()
    else:
        print(f"Unknown command: {args[0]}")
