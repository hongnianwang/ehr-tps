import numpy as np


def clean_row(row):
    """
    Cleans a time series row by removing leading and trailing zeros, replacing in-between zeros with NaN, 
    and filling NaNs using forward and backward fill.
    """
    row = row[(row != 0).cumsum() > 0][::-1][(row != 0).cumsum() > 0][::-1]
    row = row.replace(0, np.nan).ffill().bfill()

    return row.astype(int).values
