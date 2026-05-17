import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

import sys
sys.path.append("../")
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


# Train model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train_processed, y_train)


# Predictions
predictions = model.predict(X_test_processed)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"R2 Score: {r2}")


# Save model
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")


print("Model and preprocessor saved successfully.")