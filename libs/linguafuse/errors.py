import pandas as pd
from typing import Bool

def validate_columns(v: pd.DataFrame) -> Bool:
    required_columns = {'text', 'encoded_label', 'label'}
    missing_columns = required_columns - set(v.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    return True

def validate_encodings(v: pd.DataFrame) -> Bool:
    range_values = v['encoded_label'].min(), v['encoded_label'].max()
    max_len = v['label'].unique().size
    assert range_values == (0, max_len - 1), f"Encoded labels should be in the range [0, {max_len - 1}] but got {range_values}"

    return True
