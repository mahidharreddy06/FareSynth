import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from src.config import (
    AVAILABLE_ROUTES,
    SEASONS,
    SEASON_MULTIPLIERS,
    DEMAND_SCENARIOS,
    DEMAND_SCENARIO_MULTIPLIERS,
    RouteConfig,
    ProjectConfig
)

def simulate_booking_curve(
    route: RouteConfig,
    season: str,
    demand_scenario: str,
    flight_id: str,
    departure_date: datetime,
    seq_len: int = 30,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulates a 30-day realistic booking curve for a single airline flight episode.
    Follows empirical revenue management patterns:
    - Early booking window (30-15 days): Price-sensitive leisure travelers, lower fares.
    - Mid window (14-7 days): Balanced mix, gradual fare escalation.
    - Late window (6-1 days): Inelastic business travelers, rapid surge, load factor saturation.
    """
    if seed is not None:
        np.random.seed(seed)

    season_mult = SEASON_MULTIPLIERS.get(season, 1.0)
    demand_mult = DEMAND_SCENARIO_MULTIPLIERS.get(demand_scenario, 1.0)
    base_demand = season_mult * demand_mult

    capacity = route.capacity
    base_fare = route.base_fare

    # State tracking
    current_booked = int(capacity * np.random.uniform(0.05, 0.15)) # 5-15% booked early before t=30
    records = []

    for d in range(seq_len, 0, -1):
        # Progress from 0.0 (day 30) to 1.0 (day 1)
        urgency = (seq_len - d) / seq_len
        
        # Stochastic demand index with daily noise and late-window acceleration
        daily_noise = np.random.normal(0, 0.05)
        current_demand_index = max(0.2, base_demand * (1.0 + 0.3 * (urgency ** 2)) + daily_noise)
        
        current_load_factor = min(1.0, current_booked / capacity)
        available_seats = max(0, capacity - current_booked)

        # Dynamic pricing formula:
        # P = Base * Season * DemandScenario * (1 + alpha * LoadFactor^2) * (1 + beta * Urgency)
        load_factor_pressure = 1.0 + 0.8 * (current_load_factor ** 2.5)
        urgency_pressure = 1.0 + 0.45 * (urgency ** 1.8)
        
        # Target dynamic fare with micro-market fluctuation
        market_noise = np.random.normal(1.0, 0.03)
        fare_price = base_fare * (base_demand ** 0.6) * load_factor_pressure * urgency_pressure * market_noise
        fare_price = round(max(base_fare * 0.5, fare_price), 2)

        # Competitor fare: models competitor dynamic response with slight lag and variance
        comp_spread = np.random.normal(0.0, 0.06)
        competitor_fare = round(max(base_fare * 0.45, fare_price * (1.0 + comp_spread)), 2)

        # Booking probability based on price ratio, urgency, and demand
        # Price elasticity: if our fare is lower than competitor, higher conversion
        price_competitiveness = competitor_fare / fare_price
        expected_daily_rate = (capacity / seq_len) * current_demand_index * (price_competitiveness ** 0.8)
        
        # Poisson/Binomial booking arrivals
        daily_bookings = int(np.random.poisson(max(0.1, expected_daily_rate)))
        # Occasional cancellations (0-2 tickets)
        cancellations = int(np.random.binomial(current_booked, 0.015)) if current_booked > 0 else 0
        net_bookings = max(0, min(daily_bookings - cancellations, available_seats))
        
        current_booked = min(capacity, current_booked + net_bookings)
        load_factor_after = round(current_booked / capacity, 4)

        record_date = departure_date - timedelta(days=d)

        records.append({
            "flight_id": flight_id,
            "route": route.code,
            "origin": route.origin,
            "destination": route.destination,
            "season": season,
            "demand_scenario": demand_scenario,
            "capacity": capacity,
            "departure_date": departure_date.strftime("%Y-%m-%d"),
            "date": record_date.strftime("%Y-%m-%d"),
            "days_to_departure": d,
            "demand_index": round(current_demand_index, 4),
            "available_seats": available_seats,
            "load_factor": load_factor_after,
            "daily_bookings": net_bookings,
            "competitor_fare": competitor_fare,
            "fare_price": fare_price
        })

    return pd.DataFrame(records)

def generate_historical_dataset(
    flights_per_route: int = 150,
    seq_len: int = 30,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a rich historical airline fare dataset covering multiple routes,
    seasons, and demand scenarios.
    """
    all_dfs = []
    base_date = datetime(2025, 1, 1)
    flight_counter = 1

    print(f"Generating historical dataset for {len(AVAILABLE_ROUTES)} routes...")

    for route in AVAILABLE_ROUTES:
        for i in range(flights_per_route):
            season = np.random.choice(SEASONS, p=[0.25, 0.35, 0.20, 0.20])
            # Higher probability of normal/high demand
            demand_scen = np.random.choice(DEMAND_SCENARIOS, p=[0.15, 0.50, 0.25, 0.10])
            dep_date = base_date + timedelta(days=int(np.random.randint(35, 365)))
            flight_id = f"FL-{route.code.replace('-', '')}-{flight_counter:05d}"
            flight_counter += 1

            flight_df = simulate_booking_curve(
                route=route,
                season=season,
                demand_scenario=demand_scen,
                flight_id=flight_id,
                departure_date=dep_date,
                seq_len=seq_len
            )
            all_dfs.append(flight_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Historical dataset generated: {len(combined_df)} records across {flight_counter - 1} flights.")

    if output_path:
        combined_df.to_csv(output_path, index=False)
        print(f"Saved historical dataset to {output_path}")

    return combined_df

if __name__ == "__main__":
    cfg = ProjectConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df = generate_historical_dataset(flights_per_route=120, output_path=str(cfg.historical_csv))
    print(df.head())
    print("\nDataset Summary:")
    print(df.describe())
