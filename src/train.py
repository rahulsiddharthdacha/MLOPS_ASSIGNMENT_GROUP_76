import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import evaluate_model
import pandas as pd

data = fetch_california_housing(as_frame=True)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

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

        return metrics["mse"] , model_name
        
        print(f"Model: {model_name}, MSE: {metrics['mse']}, R2: {metrics['r2_score']}")



# Train and log Linear Regression model
lr_mse, lr_name = train_and_log(LinearRegression(), "LinearRegression")
dt_mse, dt_name = train_and_log(DecisionTreeRegressor(), "DecisionTreeRegressor", params={"max_depth": 5})  


best_model_name = lr_name if lr_mse < dt_mse else dt_name
best_model_mse = min(lr_mse, dt_mse)
print(f"Best Model: {best_model_name}, MSE: {best_model_mse}")