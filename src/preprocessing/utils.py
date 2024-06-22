import numpy as np
import pandas as pd


def min_max_scale_globally(sequences: np.ndarray) -> pd.Series:
    combined_sequences: np.ndarray = np.vstack(sequences)  # type: ignore[call-overload]

    min_x, min_y = combined_sequences[:, 0].min(), combined_sequences[:, 1].min()
    max_x, max_y = combined_sequences[:, 0].max(), combined_sequences[:, 1].max()

    del combined_sequences

    def min_max_scale_sequence(sequence: np.ndarray) -> np.ndarray:
        x_scaled = 2 * ((sequence[:, 0] - min_x) / (max_x - min_x)) - 1
        y_scaled = 2 * ((sequence[:, 1] - min_y) / (max_y - min_y)) - 1
        return np.column_stack((x_scaled, y_scaled))

    return pd.Series([min_max_scale_sequence(seq) for seq in sequences])


def pad_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    if arr.shape[0] < target_length:
        pad_width = target_length - arr.shape[0]
        padded_array = np.pad(
            arr, ((pad_width, 0), (0, 0)), "constant", constant_values=0
        )
        return padded_array
    return arr
