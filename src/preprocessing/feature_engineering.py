import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # Total bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"] +
        (0.5 * df["HalfBath"])
    )

    # House age
    df["HouseAge"] = 2026 - df["YearBuilt"]

    return df