import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dataset.
    """
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_data("data/raw/train.csv")
    print(df.head())