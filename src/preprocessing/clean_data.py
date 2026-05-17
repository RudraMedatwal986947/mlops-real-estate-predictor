import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Drop columns with too many missing values
    threshold = 0.4

    missing_ratio = df.isnull().mean()

    cols_to_drop = missing_ratio[missing_ratio > threshold].index

    df = df.drop(columns=cols_to_drop)

    # Fill numerical missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df