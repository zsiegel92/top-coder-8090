import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from models_and_constants import load_training_data, Input, InputOutputPair


def train():
    training_data = load_training_data()
    training_data[0]
    df_X = pd.DataFrame([point.input.model_dump() for point in training_data])
    df_Y = pd.DataFrame([point.expected_output for point in training_data])
    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=0,
        oob_score=True,
    )
    regressor.fit(df_X.values, df_Y.values.ravel())
    joblib.dump(regressor, "trained_model.pkl")
    return regressor

def load_model() -> RandomForestRegressor:
    try:
        return joblib.load("trained_model.pkl")
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
    for feature, importance in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
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
    plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Y vs Y-hat plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df_Y.values.ravel(), y=Y_hat, alpha=0.6)
    plt.plot([df_Y.values.min(), df_Y.values.max()], [df_Y.values.min(), df_Y.values.max()], 'r--', lw=2)
    plt.xlabel('True Values (Y)')
    plt.ylabel('Predicted Values (Y-hat)')
    plt.title('Y vs Y-hat')
    plt.savefig("y_vs_yhat.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # TSNE plots (using TSNE as UMAP alternative)
    print("Computing TSNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(df_X.values)
    
    # Create subplots for the three TSNE visualizations
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True Y values
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_Y.values.ravel(), cmap='viridis', alpha=0.7)
    axes[0].set_title('TSNE: True Y Values')
    axes[0].set_xlabel('TSNE 1')
    axes[0].set_ylabel('TSNE 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Predicted Y-hat values
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_hat, cmap='viridis', alpha=0.7)
    axes[1].set_title('TSNE: Predicted Y-hat Values')
    axes[1].set_xlabel('TSNE 1')
    axes[1].set_ylabel('TSNE 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Difference (Y - Y-hat)
    diff = df_Y.values.ravel() - Y_hat
    scatter3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=diff, cmap='RdBu', alpha=0.7)
    axes[2].set_title('TSNE: Prediction Error (Y - Y-hat)')
    axes[2].set_xlabel('TSNE 1')
    axes[2].set_ylabel('TSNE 2')
    plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("tsne_plots.png", dpi=300, bbox_inches='tight')
    plt.close()


def predict(input: Input) -> float:
    regressor = load_model()
    df_X = pd.DataFrame([input.model_dump()])
    Y = regressor.predict(df_X.values)
    return float(Y[0])

args = sys.argv[1:]
skip_training = args[0] == "skip-training"
if __name__ == "__main__":
    if not skip_training:
        train()
    inspect_model()