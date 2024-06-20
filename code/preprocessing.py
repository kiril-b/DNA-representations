import numpy as np
import pandas as pd


def min_max_scale_globally(sequences: np.ndarray) -> pd.Series:
    combined_sequences: np.ndarray = np.vstack(sequences)  # type: ignore[call-overload]

    min_x, min_y = combined_sequences[:, 0].min(), combined_sequences[:, 1].min()
    max_x, max_y = combined_sequences[:, 0].max(), combined_sequences[:, 1].max()

    def min_max_scale_sequence(sequence: np.ndarray) -> np.array:
        x_scaled = 2 * ((sequence[:, 0] - min_x) / (max_x - min_x)) - 1
        y_scaled = 2 * ((sequence[:, 1] - min_y) / (max_y - min_y)) - 1
        return pd.Series(np.column_stack((x_scaled, y_scaled)).tolist())

    return pd.Series([min_max_scale_sequence(seq) for seq in sequences])
