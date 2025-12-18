import pandas as pd


FILEPATH_RAW = "../data/xviii_abn_corps_soldier_readiness.csv.xlsx"
FILEPATH_CLEANED = "../data/cleaned_soldier_readiness.csv.xlsx"


"""Loads the dataset from an excel file into a pandas DataFrame."""
def load_dataset(filepath: str) -> pd.DataFrame:
  return pd.read_excel(filepath)

"""Saves the cleaned DataFrame to an Excel file."""
def save_cleaned_dataset(df: pd.DataFrame, filepath: str) -> None:
  df.to_excel(filepath, index=False)

