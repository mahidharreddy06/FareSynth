import pytest
import numpy as np
import pandas as pd
from src.config import AVAILABLE_ROUTES, FEATURE_COLUMNS
from src.data_generator import simulate_booking_curve
from src.preprocessor import AirlineDataPreprocessor
from datetime import datetime

def test_data_generation_and_shapes():
    route = AVAILABLE_ROUTES[0]
    df = simulate_booking_curve(
        route=route,
        season="Summer",
        demand_scenario="Normal",
        flight_id="TEST-001",
        departure_date=datetime(2025, 7, 1),
        seq_len=30,
        seed=42
    )
    assert len(df) == 30
    assert list(df["days_to_departure"].values) == list(range(30, 0, -1))
    assert (df["fare_price"] > 0).all()
    assert (df["load_factor"] >= 0.0).all() and (df["load_factor"] <= 1.0).all()

def test_preprocessor_fit_transform():
    route = AVAILABLE_ROUTES[0]
    dfs = [
        simulate_booking_curve(route, "Summer", "Normal", f"TEST-{i:03d}", datetime(2025, 7, 1), seq_len=30, seed=i)
        for i in range(5)
    ]
    df_combined = pd.concat(dfs, ignore_index=True)

    preprocessor = AirlineDataPreprocessor(seq_len=30)
    seqs, meta = preprocessor.fit_transform(df_combined)

    assert seqs.shape == (5, 30, len(FEATURE_COLUMNS))
    assert len(meta) == 5
    assert (seqs >= 0.0).all() and (seqs <= 1.0).all()

    # Inverse transform
    unscaled = preprocessor.inverse_transform(seqs)
    assert unscaled.shape == seqs.shape

    # Output DataFrame
    df_out = preprocessor.sequences_to_dataframe(unscaled, metadata_list=meta)
    assert len(df_out) == 150
    assert (df_out["fare_price"] > 0).all()
