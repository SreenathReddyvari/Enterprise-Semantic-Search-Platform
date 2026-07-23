import pandas as pd

from src.utils.helper import clean_text


def preprocess_documents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Content"] = df["Content"].apply(clean_text)
    df["Title"] = df["Title"].apply(clean_text)
    df = df[df["Content"].str.len() > 0].reset_index(drop=True)
    return df
