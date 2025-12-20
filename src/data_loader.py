# src/data_loader.py
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file with basic error handling.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"The file at {path} is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return df
