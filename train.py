import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
DATA_DIR = 'data/raw/housing.csv'

# data = fetch_california_housing(as_frame=True)
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = data.target

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data file not found at {DATA_DIR}. Please ensure the data is available.")

# Load the dataset
df = pd.read_csv(DATA_DIR)
TARGET_COLUMN = 'median_house_value'
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Housing Price Prediction")

def train_and_log(model, model_name, params=None):
    with mlflow.start_run(run_name=model_name):
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        mlflow.log_params(model.get_params() or params or {})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        print(f"Model: {model_name}, MSE: {metrics['mse']}, R2: {metrics['r2_score']}")

        return metrics["mse"] , model_name, model


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': mse,
        'r2_score': r2
    }


# Train and log Linear Regression model
lr_mse, lr_name, lr_model = train_and_log(LinearRegression(), "LinearRegression")
dt_mse, dt_name, dt_model = train_and_log(DecisionTreeRegressor(), "DecisionTreeRegressor", params={"max_depth": 5})


if lr_mse < dt_mse:
    best_model_name = lr_name
    best_model = lr_model
    best_model_mse = lr_mse
else:
    best_model_name = dt_name
    best_model = dt_model
    best_model_mse = dt_mse


print(f"Best Model: {best_model_name}, MSE: {best_model_mse}")


import joblib

# Save the model
joblib.dump(best_model, 'model.joblib')