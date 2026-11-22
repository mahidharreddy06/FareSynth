import sys
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    AVAILABLE_ROUTES,
    SEASONS,
    DEMAND_SCENARIOS,
    ProjectConfig,
    TimeGANConfig
)
from src.timegan import TimeGAN
from src.trainer import TimeGANTrainer
from src.preprocessor import AirlineDataPreprocessor
from src.dynamic_pricing import (
    DynamicPricingSimulator,
    FixedBasePricing,
    RuleBasedYieldManager,
    CompetitorMatchingPricer,
    AdaptiveQLearningPricer
)
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    FlightSequence,
    BookingRecord,
    PricingSimulationRequest,
    PricingSimulationResponse,
    PricingStrategyResult
)

# Global model state
cfg = ProjectConfig()
preprocessor = AirlineDataPreprocessor(seq_len=cfg.timegan.seq_len)
model: TimeGAN = None
device = "cuda" if torch.cuda.is_available() else "cpu"
frontend_dir = ROOT_DIR / "frontend"
figures_dir = cfg.figures_dir

def ensure_artifacts_loaded():
    """Ensures TimeGAN model and scaler are loaded in memory."""
    global model, preprocessor
    if not preprocessor.fitted and cfg.scaler_checkpoint.exists():
        preprocessor.load_scaler(cfg.scaler_checkpoint)

    if model is None:
        if cfg.model_checkpoint.exists():
            trainer = TimeGANTrainer.load_checkpoint(cfg.model_checkpoint, device=device)
            model = trainer.model
            print(f"TimeGAN model successfully loaded into memory on {device}.")
        else:
            model = TimeGAN(input_dim=7, hidden_dim=cfg.timegan.hidden_dim, num_layers=cfg.timegan.num_layers, noise_dim=7)
            model.to(device)
            model.eval()

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_artifacts_loaded()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="FareSynth AI: Synthetic Airline Fare Data Generator",
    description="Time-series synthetic airline fare and booking data generator powered by TimeGAN and Dynamic Pricing evaluation engine.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static figures directory so charts can be displayed in frontend
if figures_dir.exists():
    app.mount("/reports/figures", StaticFiles(directory=str(figures_dir)), name="figures")

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serves the interactive FareSynth AI web dashboard."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "status": "online",
        "message": "Frontend not found. Please visit /docs for API documentation."
    }

@app.get("/app", tags=["Frontend"])
def serve_app_alias():
    return serve_frontend()

@app.get("/api/info", tags=["General"])
def api_info():
    """Returns metadata and available endpoints."""
    return {
        "project": "FareSynth AI",
        "title": "Synthetic Airline Fare and Booking Data Generator",
        "description": "Generates realistic multivariate time-series fare, demand, and booking sequences for dynamic-pricing model testing.",
        "endpoints": {
            "web_ui": "/",
            "health": "/health",
            "routes": "/routes",
            "generate": "/generate",
            "pricing_simulation": "/pricing/simulate",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["General"])
def health_check():
    ensure_artifacts_loaded()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_fitted": preprocessor.fitted,
        "device": device,
        "framework": f"PyTorch {torch.__version__}",
        "architecture": "TimeGAN (Embedding, Recovery, Generator, Supervisor, Discriminator)"
    }

@app.get("/routes", tags=["Airline Domain"])
def list_available_routes():
    return {
        "count": len(AVAILABLE_ROUTES),
        "routes": [
            {
                "code": r.code,
                "origin": r.origin,
                "destination": r.destination,
                "base_fare": r.base_fare,
                "distance_km": r.distance_km,
                "capacity": r.capacity
            }
            for r in AVAILABLE_ROUTES
        ],
        "seasons": SEASONS,
        "demand_scenarios": DEMAND_SCENARIOS
    }

@app.post("/generate", response_model=GenerateResponse, tags=["Synthetic Generation"])
def generate_synthetic_fares(request: GenerateRequest):
    """
    Primary endpoint: Generates synthetic airline fare & booking time-series records on demand.
    Applies TimeGAN latent sampling, inverse scaling, and airline business constraints.
    """
    ensure_artifacts_loaded()
    global model, preprocessor

    # Match route or fallback
    matched_route = next((r for r in AVAILABLE_ROUTES if r.code.upper() == request.route.upper()), AVAILABLE_ROUTES[0])
    season = request.season if request.season in SEASONS else "Summer"
    demand_scen = request.demand_scenario if request.demand_scenario in DEMAND_SCENARIOS else "Normal"
    n_seq = request.num_sequences
    seq_len = request.seq_len

    if not preprocessor.fitted:
        raise HTTPException(status_code=503, detail="Scaler parameters could not be loaded.")

    if model is None:
        raise HTTPException(status_code=503, detail="TimeGAN model is not loaded.")

    model.eval()
    with torch.no_grad():
        x_fake_tensor = model.generate(n_samples=n_seq, seq_len=seq_len, device=device)
        x_fake_np = x_fake_tensor.cpu().numpy()

    # Unscale sequences back to physical units
    unscaled = preprocessor.inverse_transform(x_fake_np)

    metadata_list = [
        {
            "flight_id": f"GEN-{matched_route.code.replace('-', '')}-{i+1:04d}",
            "route": matched_route.code,
            "origin": matched_route.origin,
            "destination": matched_route.destination,
            "season": season,
            "demand_scenario": demand_scen,
            "capacity": matched_route.capacity,
            "departure_date": "2025-08-01"
        }
        for i in range(n_seq)
    ]

    df_gen = preprocessor.sequences_to_dataframe(unscaled, metadata_list=metadata_list)

    # Format structured response
    flights_output = []
    for meta in metadata_list:
        f_id = meta["flight_id"]
        flight_rows = df_gen[df_gen["flight_id"] == f_id].sort_values(by="days_to_departure", ascending=False)
        timeline = [
            BookingRecord(
                days_to_departure=int(r["days_to_departure"]),
                demand_index=float(r["demand_index"]),
                available_seats=int(r["available_seats"]),
                load_factor=float(r["load_factor"]),
                daily_bookings=int(r["daily_bookings"]),
                competitor_fare=float(r["competitor_fare"]),
                fare_price=float(r["fare_price"])
            )
            for _, r in flight_rows.iterrows()
        ]
        flights_output.append(FlightSequence(
            flight_id=f_id,
            route=meta["route"],
            origin=meta["origin"],
            destination=meta["destination"],
            season=meta["season"],
            demand_scenario=meta["demand_scenario"],
            capacity=meta["capacity"],
            departure_date=meta["departure_date"],
            timeline=timeline
        ))

    return GenerateResponse(
        status="success",
        total_flights_generated=n_seq,
        total_records=len(df_gen),
        flights=flights_output
    )

@app.post("/pricing/simulate", response_model=PricingSimulationResponse, tags=["Dynamic Pricing Testing"])
def simulate_dynamic_pricing(request: PricingSimulationRequest):
    """
    Demonstrates how synthetic fare and booking sequences are used to test
    and experiment with dynamic pricing models across surge and standard conditions.
    """
    ensure_artifacts_loaded()
    matched_route = next((r for r in AVAILABLE_ROUTES if r.code.upper() == request.route.upper()), AVAILABLE_ROUTES[0])
    base_fare = request.base_fare or matched_route.base_fare
    capacity = request.capacity or matched_route.capacity

    # Generate synthetic scenario for testing
    gen_req = GenerateRequest(
        route=matched_route.code,
        season="Summer",
        demand_scenario=request.demand_scenario,
        num_sequences=1,
        seq_len=30
    )
    gen_res = generate_synthetic_fares(gen_req)
    flight = gen_res.flights[0]

    # Convert timeline to DataFrame
    df_flight = pd.DataFrame([b.model_dump() for b in flight.timeline])

    simulator = DynamicPricingSimulator()
    sim_results = simulator.run_simulation(df_flight, base_fare=base_fare, capacity=capacity)

    strategy_results = [
        PricingStrategyResult(
            strategy=r.model_name,
            total_revenue=r.total_revenue,
            seats_sold=r.total_seats_sold,
            final_load_factor=r.final_load_factor,
            average_fare=r.average_fare,
            unsold_seats=r.unsold_seats,
            revenue_per_available_seat=r.revenue_per_available_seat
        )
        for r in sim_results
    ]

    best_strat = max(strategy_results, key=lambda x: x.total_revenue).strategy

    return PricingSimulationResponse(
        route=matched_route.code,
        demand_scenario=request.demand_scenario,
        base_fare=base_fare,
        capacity=capacity,
        results=strategy_results,
        recommended_strategy=best_strat
    )
