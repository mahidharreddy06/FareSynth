import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import FEATURE_COLUMNS, ProjectConfig

class AirlineDataPreprocessor:
    """
    Handles preprocessing, sequence windowing, scaling, and inverse-scaling
    for airline time-series sequences fed into TimeGAN.
    """
    def __init__(self, feature_columns: Optional[List[str]] = None, seq_len: int = 30):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.seq_len = seq_len
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.fitted = False
        self.scaler_params: Dict[str, Any] = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fits MinMaxScaler on numerical features and reshapes dataframe by flight_id
        into 3D sequence array of shape [N, seq_len, n_features].
        Also returns flight metadata per sequence.
        """
        # Ensure ordering by flight_id and days_to_departure descending (day 30 down to 1)
        df_sorted = df.sort_values(by=["flight_id", "days_to_departure"], ascending=[True, False]).copy()
        
        # Fit scaler on full numerical feature space
        feature_data = df_sorted[self.feature_columns].values
        scaled_features = self.scaler.fit_transform(feature_data)
        self.fitted = True

        # Store scaler parameters for export
        self.scaler_params = {
            "features": self.feature_columns,
            "min": self.scaler.data_min_.tolist(),
            "max": self.scaler.data_max_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "data_range": (self.scaler.data_max_ - self.scaler.data_min_).tolist()
        }

        # Group by flight_id into fixed sequence length
        sequences = []
        metadata = []

        flight_groups = df_sorted.groupby("flight_id")
        for flight_id, group in flight_groups:
            if len(group) == self.seq_len:
                group_scaled = self.scaler.transform(group[self.feature_columns].values)
                sequences.append(group_scaled)
                meta = {
                    "flight_id": flight_id,
                    "route": group["route"].iloc[0],
                    "season": group["season"].iloc[0],
                    "demand_scenario": group["demand_scenario"].iloc[0],
                    "capacity": int(group["capacity"].iloc[0]),
                    "origin": group["origin"].iloc[0],
                    "destination": group["destination"].iloc[0],
                    "departure_date": str(group["departure_date"].iloc[0])
                }
                metadata.append(meta)

        sequences_arr = np.array(sequences, dtype=np.float32)
        return sequences_arr, metadata

    def inverse_transform(self, scaled_sequences: np.ndarray) -> np.ndarray:
        """
        Inverts normalized [0, 1] sequences back to real-world airline units.
        scaled_sequences: shape [N, seq_len, n_features] or [seq_len, n_features]
        """
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet!")

        original_shape = scaled_sequences.shape
        if len(original_shape) == 3:
            N, S, F = original_shape
            flat = scaled_sequences.reshape(-1, F)
            inverted_flat = self.scaler.inverse_transform(flat)
            return inverted_flat.reshape(N, S, F)
        elif len(original_shape) == 2:
            return self.scaler.inverse_transform(scaled_sequences)
        else:
            raise ValueError(f"Unsupported array shape: {original_shape}")

    def save_scaler(self, file_path: Path):
        """Saves scaler state to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.scaler_params, f, indent=2)
        print(f"Scaler parameters saved to {file_path}")

    def load_scaler(self, file_path: Path):
        """Loads scaler state from JSON file."""
        with open(file_path, "r") as f:
            params = json.load(f)
        self.scaler_params = params
        self.feature_columns = params["features"]
        
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.scaler.data_min_ = np.array(params["min"], dtype=np.float64)
        self.scaler.data_max_ = np.array(params["max"], dtype=np.float64)
        self.scaler.scale_ = np.array(params["scale"], dtype=np.float64)
        self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
        self.scaler.n_samples_seen_ = 1000
        self.fitted = True
        print(f"Scaler parameters successfully loaded from {file_path}")

    def sequences_to_dataframe(
        self,
        unscaled_sequences: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Converts 3D sequence array [N, seq_len, features] into a clean structured DataFrame.
        Enforces realistic business domain constraints:
        - non-negative bookings & fares
        - integer seat counts
        - bounded load factor [0, 1]
        """
        records = []
        N, seq_len, n_feats = unscaled_sequences.shape

        for i in range(N):
            meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else {
                "flight_id": f"SYN-FL-{i+1:05d}",
                "route": "JFK-LHR",
                "season": "Summer",
                "demand_scenario": "Normal",
                "capacity": 280,
                "origin": "New York",
                "destination": "London",
                "departure_date": "2025-07-15"
            }

            capacity = meta.get("capacity", 200)

            for step in range(seq_len):
                row_vals = unscaled_sequences[i, step, :]
                row_dict = dict(zip(self.feature_columns, row_vals))

                # Apply post-generation domain constraints
                days_left = int(round(row_dict.get("days_to_departure", seq_len - step)))
                days_left = max(1, min(seq_len, days_left))
                
                demand_idx = max(0.1, round(float(row_dict.get("demand_index", 1.0)), 4))
                load_fac = min(1.0, max(0.0, round(float(row_dict.get("load_factor", 0.5)), 4)))
                avail_seats = max(0, min(capacity, int(round(float(row_dict.get("available_seats", capacity * (1 - load_fac)))))))
                daily_bkgs = max(0, int(round(float(row_dict.get("daily_bookings", 5)))))
                comp_fare = max(20.0, round(float(row_dict.get("competitor_fare", 300.0)), 2))
                fare_px = max(25.0, round(float(row_dict.get("fare_price", 320.0)), 2))

                record = {
                    "flight_id": meta.get("flight_id", f"SYN-{i+1:05d}"),
                    "route": meta.get("route", "JFK-LHR"),
                    "origin": meta.get("origin", "New York"),
                    "destination": meta.get("destination", "London"),
                    "season": meta.get("season", "Summer"),
                    "demand_scenario": meta.get("demand_scenario", "Normal"),
                    "capacity": capacity,
                    "departure_date": meta.get("departure_date", "2025-07-15"),
                    "days_to_departure": days_left,
                    "demand_index": demand_idx,
                    "available_seats": avail_seats,
                    "load_factor": load_fac,
                    "daily_bookings": daily_bkgs,
                    "competitor_fare": comp_fare,
                    "fare_price": fare_px
                }
                records.append(record)

        return pd.DataFrame(records)
