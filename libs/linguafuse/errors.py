import pandas as pd

@classmethod
def validate_columns(cls, v: pd.DataFrame) -> pd.DataFrame:
    required_columns = {'text', 'encoded_label', 'label'}
    missing_columns = required_columns - set(v.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    return v