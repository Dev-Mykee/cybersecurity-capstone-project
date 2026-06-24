import pandas as pd
import numpy as np


def load_data(filepath):
    return pd.read_csv(filepath)


def handle_missing_values(df, strategy='drop'):
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        return df.fillna(df.mean(numeric_only=True))
    return df


def remove_duplicates(df):
    return df.drop_duplicates()
