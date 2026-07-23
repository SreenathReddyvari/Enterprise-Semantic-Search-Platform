import pandas as pd


def load_documents(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Document_ID", "Title", "Category", "Content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"documents.csv missing required columns: {missing}")
    return df
