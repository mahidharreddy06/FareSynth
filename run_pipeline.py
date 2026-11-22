import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure workspace root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ProjectConfig, TimeGANConfig, AVAILABLE_ROUTES
from src.data_generator import generate_historical_dataset
from src.preprocessor import AirlineDataPreprocessor
from src.timegan import TimeGAN
from src.trainer import TimeGANTrainer
from src.validator import SyntheticDataValidator
from src.analysis import AirlinePatternAnalyzer
from src.dynamic_pricing import DynamicPricingSimulator

def run_full_pipeline(
    quick_run: bool = True,
    flights_per_route: int = 40,
    ae_epochs: int = 25,
    sup_epochs: int = 25,
    joint_epochs: int = 35
):
    """
    Executes the complete FareSynth AI workflow from raw data creation to
    TimeGAN training, synthetic generation, validation, and dynamic pricing testing.
    """
    start_time = time.time()
    cfg = ProjectConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 FARESYNTH AI: END-TO-END SYNTHETIC FARE GENERATOR PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Historical Airline Fare & Booking Data Generation
    # -------------------------------------------------------------
    print("\n[Step 1/6] Preparing Historical Airline Fare & Booking Dataset...")
    if not cfg.historical_csv.exists() or quick_run:
        df_historical = generate_historical_dataset(
            flights_per_route=flights_per_route,
            seq_len=cfg.timegan.seq_len,
            output_path=str(cfg.historical_csv)
        )
    else:
        print(f"Loading existing historical dataset from {cfg.historical_csv}")
        df_historical = pd.read_csv(cfg.historical_csv)

    print(f"Historical Dataset: {len(df_historical):,} records ({len(df_historical) // 30} flight episodes).")

    # -------------------------------------------------------------
    # Step 2: Preprocessing, Scaling & 3D Windowing
    # -------------------------------------------------------------
    print("\n[Step 2/6] Normalizing Features & Sequencing for TimeGAN...")
    preprocessor = AirlineDataPreprocessor(seq_len=cfg.timegan.seq_len)
    sequences, metadata = preprocessor.fit_transform(df_historical)
    preprocessor.save_scaler(cfg.scaler_checkpoint)
    print(f"Sequences shape: {sequences.shape} (N_flights={sequences.shape[0]}, Days={sequences.shape[1]}, Features={sequences.shape[2]})")

    # -------------------------------------------------------------
    # Step 3: TimeGAN Model Training
    # -------------------------------------------------------------
    print("\n[Step 3/6] Initializing & Training TimeGAN Model...")
    timegan_cfg = TimeGANConfig(
        seq_len=cfg.timegan.seq_len,
        hidden_dim=32,
        num_layers=2,
        batch_size=32 if quick_run else 64,
        noise_dim=7,
        learning_rate=1e-3
    )

    model = TimeGAN(
        input_dim=7,
        hidden_dim=timegan_cfg.hidden_dim,
        num_layers=timegan_cfg.num_layers,
        noise_dim=timegan_cfg.noise_dim
    )
    trainer = TimeGANTrainer(model=model, config=timegan_cfg)
    trainer.fit(
        sequences=sequences,
        ae_epochs=ae_epochs,
        sup_epochs=sup_epochs,
        joint_epochs=joint_epochs
    )
    trainer.save_checkpoint(cfg.model_checkpoint)

    # -------------------------------------------------------------
    # Step 4: Synthetic Airline Fare Generation
    # -------------------------------------------------------------
    print("\n[Step 4/6] Sampling Synthetic Airline Fare & Booking Sequences...")
    num_synthetic_flights = min(150, len(metadata))
    model.eval()
    x_fake_scaled = model.generate(n_samples=num_synthetic_flights, seq_len=cfg.timegan.seq_len)
    x_fake_np = x_fake_scaled.cpu().numpy()

    # Inverse transform
    unscaled_synthetic = preprocessor.inverse_transform(x_fake_np)

    # Select representative metadata
    synthetic_metadata = metadata[:num_synthetic_flights]
    df_synthetic = preprocessor.sequences_to_dataframe(unscaled_synthetic, metadata_list=synthetic_metadata)
    df_synthetic.to_csv(cfg.synthetic_csv, index=False)
    print(f"Generated {len(df_synthetic):,} synthetic records saved to {cfg.synthetic_csv}")

    # -------------------------------------------------------------
    # Step 5: Validation & Quality Evaluation
    # -------------------------------------------------------------
    print("\n[Step 5/6] Validating Synthetic Data & Generating Report...")
    validator = SyntheticDataValidator(real_df=df_historical, synthetic_df=df_synthetic)
    validator.generate_visualizations(output_dir=cfg.figures_dir)
    report_text = validator.create_validation_report(output_path=cfg.reports_dir / "synthetic_fare_validation_report.md")

    # Analytical patterns
    analyzer = AirlinePatternAnalyzer(df_synthetic)
    analyzer.generate_analytical_visualizations(output_dir=cfg.figures_dir)

    # -------------------------------------------------------------
    # Step 6: Dynamic Pricing Experimentation & Testing
    # -------------------------------------------------------------
    print("\n[Step 6/6] Testing Dynamic Pricing Strategies on Synthetic Data...")
    pricing_simulator = DynamicPricingSimulator()
    comparison_df = pricing_simulator.compare_on_scenarios(df_synthetic)
    print("\nDynamic Pricing Simulation Results across Scenarios:")
    print(comparison_df.to_string(index=False))

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"🎉 PIPELINE EXECUTION COMPLETED IN {elapsed}s!")
    print(f"📁 Historical Dataset: {cfg.historical_csv}")
    print(f"📁 Synthetic Dataset:  {cfg.synthetic_csv}")
    print(f"📁 Model Checkpoint:   {cfg.model_checkpoint}")
    print(f"📁 Validation Report:  {cfg.reports_dir / 'synthetic_fare_validation_report.md'}")
    print(f"📊 Figures Directory:  {cfg.figures_dir}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FareSynth AI Pipeline Runner")
    parser.add_argument("--quick", action="store_true", default=True, help="Run with optimized epochs for swift training")
    parser.add_argument("--full", action="store_true", help="Run full extended training epochs")
    args = parser.parse_args()

    if args.full:
        run_full_pipeline(quick_run=False, flights_per_route=60, ae_epochs=50, sup_epochs=50, joint_epochs=70)
    else:
        run_full_pipeline(quick_run=True, flights_per_route=35, ae_epochs=20, sup_epochs=20, joint_epochs=30)
