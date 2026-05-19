import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

from src.preprocessing.pipeline import create_pipeline


# Load dataset
df = pd.read_csv("data/processed/cleaned_train.csv")


# Split features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# Identify column types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# Create preprocessing pipeline
preprocessor = create_pipeline(
    numeric_features,
    categorical_features
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# MLflow experiment
mlflow.set_experiment("Real Estate Price Prediction")


with mlflow.start_run():

    # Model parameters
    params = {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 8,
        "random_state": 42
    }

    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = XGBRegressor(**params)

    model.fit(X_train_processed, y_train)

    # Predictions
    predictions = model.predict(X_test_processed)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Save artifacts
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Log model
    mlflow.sklearn.log_model(
    sk_model=model,
    name="xgboost-model"
)

    print("Model tracking complete.")