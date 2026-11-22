import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ProjectConfig, AVAILABLE_ROUTES
from src.timegan import TimeGAN
from src.trainer import TimeGANTrainer
from src.preprocessor import AirlineDataPreprocessor
from src.dynamic_pricing import DynamicPricingSimulator

def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"✈️  {title.upper()}")
    print("=" * 75)

def main():
    cfg = ProjectConfig()

    print_header("FareSynth AI: Live Project Demonstration")
    print("Project: Synthetic Airline Fare Data Generator for Dynamic Pricing Models")
    print("Technology: Python 3.13 | PyTorch 2.11 (TimeGAN) | Pandas | FastAPI | Scikit-Learn\n")

    time.sleep(1)

    # 1. Inspect Datasets
    print_header("1. Historical & Synthetic Dataset Showcase")
    if cfg.historical_csv.exists() and cfg.synthetic_csv.exists():
        df_real = pd.read_csv(cfg.historical_csv)
        df_syn = pd.read_csv(cfg.synthetic_csv)
        print(f"📊 Historical Dataset Records: {len(df_real):,} across {df_real['flight_id'].nunique()} flight episodes")
        print(f"✨ Synthetic Dataset Records:  {len(df_syn):,} across {df_syn['flight_id'].nunique()} flight episodes")
        print("\nSample Synthetic Records (JFK-LHR 30-Day Trajectory):")
        sample_flight = df_syn[df_syn["route"] == "JFK-LHR"]["flight_id"].iloc[0]
        subset = df_syn[df_syn["flight_id"] == sample_flight][
            ["days_to_departure", "fare_price", "competitor_fare", "available_seats", "load_factor", "daily_bookings"]
        ].head(6)
        print(subset.to_string(index=False))
    else:
        print("Datasets not found. Run 'python run_pipeline.py' first.")

    time.sleep(1)

    # 2. Live TimeGAN Generation
    print_header("2. On-Demand Synthetic Sequence Generation (Trained TimeGAN)")
    print("Loading checkpoint 'models/timegan_fare_model.pt'...")
    preprocessor = AirlineDataPreprocessor(seq_len=cfg.timegan.seq_len)
    preprocessor.load_scaler(cfg.scaler_checkpoint)
    trainer = TimeGANTrainer.load_checkpoint(cfg.model_checkpoint, device="cpu")

    print("\nGenerating 1 new synthetic flight episode for SFO-NRT (Summer, Surge Demand)...")
    fake_scaled = trainer.model.generate(n_samples=1, seq_len=30, device="cpu").numpy()
    unscaled = preprocessor.inverse_transform(fake_scaled)

    sample_meta = [{
        "flight_id": "DEMO-SFO-NRT-001",
        "route": "SFO-NRT",
        "origin": "San Francisco",
        "destination": "Tokyo",
        "season": "Summer",
        "demand_scenario": "Surge",
        "capacity": 300,
        "departure_date": "2025-08-15"
    }]
    df_live = preprocessor.sequences_to_dataframe(unscaled, sample_meta)
    print("\nGenerated Booking Window Sample (Last 5 Days to Departure):")
    print(df_live[["days_to_departure", "fare_price", "load_factor", "available_seats", "daily_bookings"]].tail(5).to_string(index=False))

    time.sleep(1)

    # 3. Dynamic Pricing Model Simulation
    print_header("3. Dynamic Pricing Experimentation on Synthetic Data")
    print("Testing 4 Pricing Strategies on the generated SFO-NRT Surge scenario:")
    simulator = DynamicPricingSimulator()
    pricing_res = simulator.run_simulation(df_live, base_fare=850.0, capacity=300)

    res_table = []
    for r in pricing_res:
        res_table.append({
            "Strategy": r.model_name,
            "Total Revenue": f"${r.total_revenue:,.2f}",
            "Seats Sold": f"{r.total_seats_sold}/300",
            "Final Load Factor": f"{r.final_load_factor * 100:.1f}%",
            "Avg Fare": f"${r.average_fare:.2f}",
            "Revenue/Seat": f"${r.revenue_per_available_seat:.2f}"
        })
    print(pd.DataFrame(res_table).to_string(index=False))

    time.sleep(1)

    # 4. How to show the API & Visualizations
    print_header("4. How to Open Interactive FastAPI and Visual Reports")
    print("To launch the live interactive Web API:")
    print("  Command:  uvicorn app.main:app --reload")
    print("  Swagger:  http://127.0.0.1:8000/docs")
    print("\nTo inspect Visual Analytics & Validation plots:")
    print(f"  Folder:   {cfg.figures_dir}")
    print("  - distributions_comparison.png (Marginal distributions overlap)")
    print("  - pca_latent_overlap.png        (2D PCA latent trajectory)")
    print("  - temporal_fare_trajectory.png  (30-day dynamic price escalation)")
    print("  - demand_scenario_curves.png    (Surge vs Low demand comparison)")
    print("\n" + "=" * 75)
    print("✅ DEMONSTRATION READY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
