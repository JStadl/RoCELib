import numpy as np
import pandas as pd


def euclidean(x: pd.DataFrame, c: pd.DataFrame) -> float:
    """
    Computes the Euclidean distance between two DataFrames.

    @param x: pd.DataFrame, the first DataFrame to compare.
    @param c: pd.DataFrame, the second DataFrame to compare.
    @return: float, the Euclidean distance between the two DataFrames.
    """

    # Validate input types
    if not isinstance(x, pd.DataFrame) or not isinstance(c, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    # Ensure both DataFrames contain only numeric values
    if not np.issubdtype(x.values.dtype, np.number) or not np.issubdtype(c.values.dtype, np.number):
        raise ValueError("DataFrames must contain only numeric values.")

    return np.sqrt(np.sum((x.values - c.values) ** 2))


def manhattan(x: pd.DataFrame, c: pd.DataFrame) -> float:
    """
    Computes the Manhattan distance between two DataFrames.

    @param x: pd.DataFrame, the first DataFrame to compare.
    @param c: pd.DataFrame, the second DataFrame to compare.
    @return: float, the Manhattan distance between the two DataFrames.
    """

    if not isinstance(x, pd.DataFrame) or not isinstance(c, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    if not np.issubdtype(x.values.dtype, np.number) or not np.issubdtype(c.values.dtype, np.number):
        raise ValueError("DataFrames must contain only numeric values.")

    return np.sum(np.abs(x.values - c.values)).item()
