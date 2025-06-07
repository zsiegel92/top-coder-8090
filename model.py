import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from models_and_constants import load_training_data, Input, InputOutputPair


def train_model(hyperparams: Optional[Dict[str, Any]] = None) -> RandomForestRegressor:
    """Train a RandomForest model with given hyperparameters."""
    training_data = load_training_data()
    df_X = pd.DataFrame([point.input.model_dump() for point in training_data])
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    # Default hyperparameters
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

    # Only set oob_score if bootstrap is True
    if default_params.get("bootstrap", True):
        default_params["oob_score"] = True

    regressor = RandomForestRegressor(**default_params)
    regressor.fit(df_X.values, df_Y.values.ravel())
    return regressor


def train():
    """Legacy train function for backward compatibility."""
    regressor = train_model()
    joblib.dump(regressor, "trained_model.pkl")
    return regressor


def load_model() -> RandomForestRegressor:
    try:
        return joblib.load("best_trained_model.pkl")
    except FileNotFoundError as e:
        raise e
        # return train()


def inspect_model():
    regressor = load_model()
    tree_to_plot = regressor.estimators_[0]
    training_data = load_training_data()
    df_X = pd.DataFrame([point.input.model_dump() for point in training_data])
    df_Y = pd.DataFrame([point.expected_output for point in training_data])
    Y_hat = regressor.predict(df_X.values)
    oob_score = regressor.oob_score_
    r2 = r2_score(df_Y.values, Y_hat)
    MSE = mean_squared_error(df_Y.values, Y_hat)
    print(f"oob_score: {oob_score}")
    print(f"r2: {r2}")
    print(f"MSE: {MSE}")

    # Try showing feature importance first
    importances = regressor.feature_importances_
    feature_importance_dict = dict(zip(df_X.columns, importances))
    print("Feature Importances:")
    for feature, importance in sorted(
        feature_importance_dict.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feature}: {importance:.4f}")

    # Try to make tree more interpretable with limited depth
    plt.figure(figsize=(25, 15))
    plot_tree(
        tree_to_plot,
        feature_names=df_X.columns.tolist(),
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4,  # Slightly deeper for more insight
    )
    plt.title("Decision Tree from Random Forest (Limited Depth)")
    plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Y vs Y-hat plot
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

    # TSNE plots (using TSNE as UMAP alternative)
    print("Computing TSNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(df_X.values)

    # Create subplots for the three TSNE visualizations
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # True Y values
    scatter1 = axes[0].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=df_Y.values.ravel(), cmap="viridis", alpha=0.7
    )
    axes[0].set_title("TSNE: True Y Values")
    axes[0].set_xlabel("TSNE 1")
    axes[0].set_ylabel("TSNE 2")
    plt.colorbar(scatter1, ax=axes[0])

    # Predicted Y-hat values
    scatter2 = axes[1].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=Y_hat, cmap="viridis", alpha=0.7
    )
    axes[1].set_title("TSNE: Predicted Y-hat Values")
    axes[1].set_xlabel("TSNE 1")
    axes[1].set_ylabel("TSNE 2")
    plt.colorbar(scatter2, ax=axes[1])

    # Difference (Y - Y-hat)
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


def train_model_on_data(
    X: np.ndarray, y: np.ndarray, hyperparams: Optional[Dict[str, Any]] = None
) -> RandomForestRegressor:
    """Train model on specific data arrays."""
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

    # Only set oob_score if bootstrap is True
    if default_params.get("bootstrap", True):
        default_params["oob_score"] = True

    regressor = RandomForestRegressor(**default_params)
    regressor.fit(X, y)
    return regressor


def cross_validate_model(
    hyperparams: Dict[str, Any], k_folds: int = 5
) -> Dict[str, Any]:
    """Perform k-fold cross-validation on a model with given hyperparameters."""
    training_data = load_training_data()
    df_X = pd.DataFrame([point.input.model_dump() for point in training_data])
    df_Y = pd.DataFrame([point.expected_output for point in training_data])

    X, y = df_X.values, df_Y.values.ravel()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_scores = []
    fold_mse = []
    fold_r2 = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model on fold
        model = train_model_on_data(X_train, y_train, hyperparams)

        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_mse.append(mse)
        fold_r2.append(r2)
        fold_scores.append(r2)  # Use R² as primary score

    return {
        "mean_cv_score": np.mean(fold_scores),
        "std_cv_score": np.std(fold_scores),
        "mean_mse": np.mean(fold_mse),
        "std_mse": np.std(fold_mse),
        "mean_r2": np.mean(fold_r2),
        "std_r2": np.std(fold_r2),
        "hyperparams": hyperparams,
    }


def generate_hyperparameter_grid(quick_search: bool = False) -> List[Dict[str, Any]]:
    """Generate grid of hyperparameters to search over."""
    if quick_search:
        # Smaller grid for testing
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True],
        }
    else:
        # Full grid
        param_grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def hyperparameter_search(
    k_folds: int = 5, max_experiments: int = 50, quick_search: bool = False
) -> pd.DataFrame:
    """Perform hyperparameter search using cross-validation."""
    grid = generate_hyperparameter_grid(quick_search=quick_search)

    # Limit experiments if grid is too large
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

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by mean CV score (R²) descending
    df_results = df_results.sort_values("mean_cv_score", ascending=False)

    return df_results


def run_hyperparameter_experiment() -> None:
    """Run complete hyperparameter search and save results."""
    print("Starting hyperparameter search...")
    results_df = hyperparameter_search(k_folds=5, max_experiments=50)

    # Save results to CSV
    results_df.to_csv("hyperparameter_results.csv", index=False)
    print("Results saved to hyperparameter_results.csv")

    # Print top 5 results
    print("\nTop 5 hyperparameter combinations:")
    print("=" * 80)
    for i, (_, row) in enumerate(results_df.head().iterrows()):
        print(f"\nRank {i+1}:")
        print(f"  Mean CV R²: {row['mean_cv_score']:.4f} ± {row['std_cv_score']:.4f}")
        print(f"  Mean MSE: {row['mean_mse']:.2f} ± {row['std_mse']:.2f}")
        print(f"  Hyperparams: {row['hyperparams']}")

    # Train final model with best hyperparameters
    best_params = results_df.iloc[0]["hyperparams"]
    print(f"\nTraining final model with best hyperparameters: {best_params}")
    best_model = train_model(best_params)
    joblib.dump(best_model, "best_trained_model.pkl")
    print("Best model saved as best_trained_model.pkl")


def predict(input: Input) -> float:
    regressor = load_model()
    df_X = pd.DataFrame([input.model_dump()])
    Y = regressor.predict(df_X.values)
    return float(Y[0])


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        # Default behavior: train and inspect
        train()
        inspect_model()
    elif args[0] == "skip-training":
        # Skip training, just inspect
        inspect_model()
    elif args[0] == "hyperparameter-search":
        # Run hyperparameter search
        run_hyperparameter_experiment()
    elif args[0] == "quick-search":
        # Run quick hyperparameter search for testing
        print("Running quick hyperparameter search...")
        results_df = hyperparameter_search(
            k_folds=3, max_experiments=12, quick_search=True
        )
        print(f"\nCompleted {len(results_df)} experiments")
        print("\nTop 3 results:")
        print(results_df.head(3)[["mean_cv_score", "mean_mse", "hyperparams"]])
    elif args[0] == "test-cv":
        # Test cross-validation with default params
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
    else:
        print("Usage:")
        print("  python model.py                       # Train and inspect model")
        print("  python model.py skip-training         # Skip training, just inspect")
        print(
            "  python model.py hyperparameter-search # Run full hyperparameter search"
        )
        print(
            "  python model.py quick-search          # Run quick hyperparameter search"
        )
        print("  python model.py test-cv               # Test cross-validation")
