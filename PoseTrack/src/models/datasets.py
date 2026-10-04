import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BaseSequenceDataset(Dataset):
    """
    Base sequence dataset for sliding-window multi-joint pose models.
    Supports dynamic mapping of column names to support both old convention
    (pitch, roll, yaw) and anatomical convention (flexion, abduction, rotation).
    """
    def __init__(
        self,
        csv_path: str,
        feature_cols: list[str],
        target_cols: list[str],
        sequence_length: int = 30,
        step_size: int = 5,
        scaler_stats: dict | None = None,
        normalize: bool = False
    ):
        df = pd.read_csv(csv_path)
        available = df.columns.tolist()

        # Helper to map columns if the specified column is not in CSV
        def map_col(col):
            if col in available:
                return col
            # Try mapping pitch -> flexion, roll -> abduction, yaw -> rotation
            replacements = {
                "pitch": "flexion",
                "roll": "abduction",
                "yaw": "rotation",
                "flexion": "pitch",
                "abduction": "roll",
                "rotation": "yaw"
            }
            for k, v in replacements.items():
                if k in col:
                    new_col = col.replace(k, v)
                    if new_col in available:
                        return new_col
            return col

        feature_cols_mapped = [map_col(c) for c in feature_cols]
        target_cols_mapped = [map_col(c) for c in target_cols]

        # Verify columns exist
        missing_feats = [c for c in feature_cols_mapped if c not in available]
        missing_targets = [c for c in target_cols_mapped if c not in available]
        if missing_feats or missing_targets:
            raise ValueError(f"Missing columns in dataset CSV: features={missing_feats}, targets={missing_targets}")

        raw_inputs = df[feature_cols_mapped].values.astype(np.float32)
        raw_targets = df[target_cols_mapped].values.astype(np.float32)
        
        self.normalize = normalize
        if self.normalize:
            if scaler_stats is None:
                self.col_min = raw_inputs.min(axis=0)
                self.col_max = raw_inputs.max(axis=0)
            else:
                self.col_min = np.array(scaler_stats["col_min"], dtype=np.float32)
                self.col_max = np.array(scaler_stats["col_max"], dtype=np.float32)
            
            rng = self.col_max - self.col_min
            rng[rng == 0] = 1.0
            self.inputs = 2.0 * (raw_inputs - self.col_min) / rng - 1.0
        else:
            self.inputs = raw_inputs
            self.col_min = None
            self.col_max = None
            
        self.targets = raw_targets
        self.sequence_length = sequence_length
        self.step_size = step_size
        
        self.samples = []
        n_frames = len(df)
        for start_idx in range(0, n_frames - sequence_length + 1, step_size):
            end_idx = start_idx + sequence_length
            x_seq = self.inputs[start_idx:end_idx]
            y_seq = self.targets[start_idx:end_idx]
            self.samples.append((x_seq, y_seq))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
        
    def scaler_dict(self) -> dict:
        if self.normalize:
            return {
                "col_min": self.col_min.tolist(),
                "col_max": self.col_max.tolist(),
            }
        return {}
