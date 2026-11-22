import pytest
import numpy as np
import pandas as pd
from src.config import AVAILABLE_ROUTES
from src.data_generator import simulate_booking_curve
from src.dynamic_pricing import (
    FixedBasePricing,
    RuleBasedYieldManager,
    CompetitorMatchingPricer,
    AdaptiveQLearningPricer,
    DynamicPricingSimulator
)
from datetime import datetime

def test_pricing_strategies_logic():
    fixed = FixedBasePricing()
    yield_mgr = RuleBasedYieldManager()
    comp = CompetitorMatchingPricer()
    q_agent = AdaptiveQLearningPricer()

    base_fare = 300.0

    # Test baseline
    assert fixed.quote_fare(base_fare, 10, 0.5, 310.0, 1.0) == 300.0

    # Low load factor early on -> discount
    early_fare = yield_mgr.quote_fare(base_fare, days_to_departure=25, load_factor=0.2, competitor_fare=300.0, demand_index=1.0)
    assert early_fare < base_fare

    # High load factor near departure -> premium
    late_fare = yield_mgr.quote_fare(base_fare, days_to_departure=2, load_factor=0.92, competitor_fare=450.0, demand_index=1.2)
    assert late_fare > base_fare

    # Competitor matching pricer undercuts when load factor is low
    undercut_fare = comp.quote_fare(base_fare, 15, 0.40, competitor_fare=350.0, demand_index=1.0)
    assert undercut_fare < 350.0

def test_dynamic_pricing_simulation():
    route = AVAILABLE_ROUTES[0]
    flight_df = simulate_booking_curve(
        route=route,
        season="Summer",
        demand_scenario="High",
        flight_id="SIM-001",
        departure_date=datetime(2025, 8, 1),
        seq_len=30,
        seed=101
    )

    sim = DynamicPricingSimulator()
    results = sim.run_simulation(flight_df, base_fare=route.base_fare, capacity=route.capacity)

    assert len(results) == 4
    for r in results:
        assert r.total_revenue > 0
        assert r.total_seats_sold <= route.capacity
        assert 0.0 <= r.final_load_factor <= 1.0
        assert r.average_fare > 0
