from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

'''
Constants.
'''

# There are 15 sensor channels (5 magnetometers x 3 axes)
# Builds list ["mag1_x", "mag1_y", "mag1__z", "mag2_x",..., "mag5_z"]
MAG_COLS = []
sensors = range(1,6)
axes = "xyz"
for s in sensors:
    for a in axes:
        MAG_COLS.append(f"mag{s}_{a}")

SLIP_COL = "slip"
TIME_COL = "log_time"
HF_REPO_ID = "pollen-robotics/anyskin_slip_detection"

'''
Loads dataset from huggingface.
Produces and stacks DataFrames for all 16 objects with added 'object' column.
'''

def load_from_hub(cache_dir: str | None = None) -> pd.DataFrame:

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = []
    for file in api.list_repo_files(HF_REPO_ID, repo_type = "dataset"):
        if file.startswith("data/") and file.endswith(".csv"):
            files.append(file)

    if not files:
        raise RuntimeError("No CSV files found under data/ in HF repo.")

    data_frames = []
    for file in files:
        local_path = hf_hub_download(
            HF_REPO_ID, file, repo_type = "dataset", cache_dir = cache_dir
        )
        obj_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(local_path)
        df["object"] = obj_name
        data_frames.append(df)

    # Stacks all per-object DataFrames (row-wise)
    # Restarts row index count as 0,1,...
    return pd.concat(data_frames, ignore_index = True)

'''
Loads dataset from local directory.
Produces and stacks DataFrames for all 16 objects with added 'object' column.
'''

def load_local_dir(directory: str) -> pd.DataFrame:

    all_files = os.listdir(directory)
    csv_files = []
    for f in all_files:
        if f.endswith(".csv"):
            csv_files.append(f)

    if not csv_files:
        raise FileNotFoundError("No CSV files found in local directory.")

    # Maintain same alphanumeric order by sorting
    sorted_files = sorted(csv_files)

    data_frames = []
    for file in sorted_files:
        local_path = os.path.join(directory, file)
        obj_name = os.path.splitext(file)[0]
        df = pd.read_csv(local_path)
        df["object"] = obj_name
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index = True)

'''
Splits each object's data into separate trials.
Cuts whenever there is a big time gap (gripper reset).
'''

def segment_on_time_gap(df: pd.DataFrame, max_gap_ms: float = 50.0) -> list[pd.DataFrame]:

    segments = []

    for object, group in df.groupby("object", sort = False):
        group = group.sort_values(TIME_COL).reset_index(drop = True)
        time = pd.to_datetime(group[TIME_COL])
        gap_ms = time.diff().dt.total_seconds() * 1000
        # Produces something like [0,0,0,0,1,1,1,2,2,...]
        # To represent each trial for an object
        seg_id = (gap_ms > max_gap_ms).cumsum().fillna(0)
        for _, seg in group.groupby(seg_id):
            if len(seg) > 1:
                segments.append(seg.reset_index(drop = True))

    return segments

## Windowing + Pre-processing ##

'''
High-pass filter that suppresses slow, steady components of signal
y[i] = alpha * x[i] - (1 - alpha) * y[i-1]
'''

def high_pass_filter(window: np.ndarray, alpha: float = 0.2) -> np.ndarray:

    y = np.zeros_like(window, dtype = np.float64)
    y[0] = alpha * window[0]
    for i in range(1, window.shape[0]):
        y[i] = alpha * window[i] - (1 - alpha) * y[i - 1]
    # Return filtered version of window
    return y

'''
Converts raw readings over time to frequency-magnitude values.
'''

def fft_features(window: np.ndarray) -> np.ndarray:
    """
    Per-channel FFT magnitude, truncated/padded to the same length T as the
    input window so it can be stacked with the time-domain block.
    window: (T, C) -> (T, C)
    """
    T = window.shape[0]
    spec = np.abs(np.fft.fft(window, axis = 0))
    return spec[:T]

'''
Input is per-trial DataFrames produced by segment_on_time_gap.
16 rows go in each windows, step_size/stride is 1.
Walks through each segment, slides window, builds feature block + 
slip label + record of where object came from.
'''

def make_windows(
    segments: list[pd.DataFrame],
    window: int = 16,
    stride: int = 1,
    use_fft: bool = True,
    use_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    X, y, groups = [], [], []

    # Windows are never built across a trial boundary
    # as a segment passes through segment_on_time_gap
    # and never contains rows from two different trials

    for seg in segments:
        arr = seg[MAG_COLS].to_numpy(dtype = np.float64)
        labels = seg[SLIP_COL].to_numpy()
        obj = seg["object"].iloc[0] # Object name
        n = arr.shape[0] # No. of rows in seg
        for start in range(0, n - window + 1, stride):
            raw = arr[start:start + window]
            time_block = high_pass_filter(raw) if use_filter else raw
            if use_fft:
                freq_block = fft_features(raw)
                # First 16 rows freq domain, next 16 time domain
                block = np.concatenate([freq_block, time_block], axis = 0)
            else:
                block = time_block
            X.append(block)
            # (start + window - 1) is index of last row covered by window
            # labels[start + window - 1] describes slip/no-slip at end
            y.append(labels[start + window - 1])
            # To group windows by object they comes from
            groups.append(obj)

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype = np.int64)
    groups = np.asarray(groups)
    return X, y, groups


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def build_dataset(
    df: pd.DataFrame,
    window: int = 16,
    stride: int = 1,
    max_gap_ms: float = 50.0,
    use_fft: bool = True,
    use_filter: bool = True,
) -> DatasetBundle:
    segments = segment_on_time_gap(df, max_gap_ms = max_gap_ms)
    X, y, groups = make_windows(
        segments, window=window, stride=stride, use_fft=use_fft, use_filter=use_filter
    )
    return DatasetBundle(X = X, y = y, groups = groups)