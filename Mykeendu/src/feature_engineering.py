import pandas as pd
import numpy as np


def encode_categorical(df, columns):
    return pd.get_dummies(df, columns=columns)


def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df
