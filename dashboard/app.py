import streamlit as st
import requests


st.title("🏠 Real Estate Price Predictor")

st.write("Enter house details below")


# Inputs
overall_qual = st.slider("Overall Quality", 1, 10, 5)

gr_liv_area = st.number_input(
    "Ground Living Area",
    min_value=500,
    max_value=5000,
    value=1500
)

garage_cars = st.slider(
    "Garage Capacity",
    0,
    5,
    2
)

total_bsmt_sf = st.number_input(
    "Basement Area",
    min_value=0,
    max_value=3000,
    value=800
)

full_bath = st.slider(
    "Full Bathrooms",
    0,
    5,
    2
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026,
    value=2000
)


# Predict button
if st.button("Predict Price"):

    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    prediction = response.json()

    st.success(
        f"Predicted House Price: ${prediction['predicted_price']:,.2f}"
    )