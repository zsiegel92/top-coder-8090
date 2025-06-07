import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
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
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_to_plot,
        feature_names=df_X.columns.tolist(),
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Decision Tree from Random Forest")
    plt.savefig("decision_tree.png")
    # TODO: show UMAP of 3d data colored by error


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