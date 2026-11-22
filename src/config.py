import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

@dataclass
class RouteConfig:
    code: str
    origin: str
    destination: str
    base_fare: float
    distance_km: int
    capacity: int

AVAILABLE_ROUTES = [
    RouteConfig("JFK-LHR", "New York", "London", 650.0, 5550, 280),
    RouteConfig("LAX-ORD", "Los Angeles", "Chicago", 280.0, 2800, 180),
    RouteConfig("SFO-NRT", "San Francisco", "Tokyo", 850.0, 8280, 300),
    RouteConfig("DXB-LHR", "Dubai", "London", 550.0, 5500, 260),
    RouteConfig("DEL-BOM", "Delhi", "Mumbai", 110.0, 1150, 180),
    RouteConfig("ATL-MIA", "Atlanta", "Miami", 190.0, 960, 160),
]

SEASONS = ["Spring", "Summer", "Fall", "Winter"]
SEASON_MULTIPLIERS = {
    "Spring": 1.05,
    "Summer": 1.35,  # Peak vacation season
    "Fall": 0.95,
    "Winter": 1.15   # Holiday travel
}

DEMAND_SCENARIOS = ["Low", "Normal", "High", "Surge"]
DEMAND_SCENARIO_MULTIPLIERS = {
    "Low": 0.70,
    "Normal": 1.00,
    "High": 1.30,
    "Surge": 1.65
}

# Features modeled sequentially in TimeGAN
FEATURE_COLUMNS = [
    "days_to_departure",
    "demand_index",
    "available_seats",
    "load_factor",
    "daily_bookings",
    "competitor_fare",
    "fare_price"
]

@dataclass
class TimeGANConfig:
    seq_len: int = 30
    hidden_dim: int = 32
    num_layers: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 1.0  # Weight for unsupervised vs supervised loss
    autoencoder_epochs: int = 60
    supervisor_epochs: int = 60
    joint_epochs: int = 100
    noise_dim: int = 7
    device: str = "cpu"  # Torch device

@dataclass
class ProjectConfig:
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    historical_csv: Path = DATA_DIR / "historical_airline_fares.csv"
    synthetic_csv: Path = DATA_DIR / "synthetic_airline_fares.csv"
    model_checkpoint: Path = MODELS_DIR / "timegan_fare_model.pt"
    scaler_checkpoint: Path = MODELS_DIR / "scaler_params.json"
    timegan: TimeGANConfig = field(default_factory=TimeGANConfig)
