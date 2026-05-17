import joblib
import pandas as pd

from fastapi import FastAPI

from api.schema import HouseData


# Load model and preprocessor
model = joblib.load("models/xgboost_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


app = FastAPI(
    title="Real Estate Price Predictor API"
)


@app.get("/")
def home():
    return {
        "message": "API is running"
    }


@app.post("/predict")
def predict(data: HouseData):

    input_data = pd.DataFrame([data.dict()])

    # Add missing columns
    required_columns = preprocessor.feature_names_in_

    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[required_columns]

    # Transform
    processed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }