from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GenerateRequest(BaseModel):
    route: Optional[str] = Field("JFK-LHR", description="Route code (e.g. JFK-LHR, LAX-ORD, SFO-NRT, DXB-LHR, DEL-BOM, ATL-MIA)")
    season: Optional[str] = Field("Summer", description="Travel season (Spring, Summer, Fall, Winter)")
    demand_scenario: Optional[str] = Field("Normal", description="Demand condition (Low, Normal, High, Surge)")
    num_sequences: int = Field(1, ge=1, le=50, description="Number of flight episodes to generate")
    seq_len: int = Field(30, ge=7, le=60, description="Booking horizon length in days")

class BookingRecord(BaseModel):
    days_to_departure: int
    demand_index: float
    available_seats: int
    load_factor: float
    daily_bookings: int
    competitor_fare: float
    fare_price: float

class FlightSequence(BaseModel):
    flight_id: str
    route: str
    origin: str
    destination: str
    season: str
    demand_scenario: str
    capacity: int
    departure_date: str
    timeline: List[BookingRecord]

class GenerateResponse(BaseModel):
    status: str
    total_flights_generated: int
    total_records: int
    flights: List[FlightSequence]

class PricingSimulationRequest(BaseModel):
    route: Optional[str] = Field("JFK-LHR", description="Target route code")
    demand_scenario: Optional[str] = Field("Surge", description="Scenario to stress test")
    base_fare: Optional[float] = Field(650.0, description="Base flight fare")
    capacity: Optional[int] = Field(280, description="Aircraft seat capacity")

class PricingStrategyResult(BaseModel):
    strategy: str
    total_revenue: float
    seats_sold: int
    final_load_factor: float
    average_fare: float
    unsold_seats: int
    revenue_per_available_seat: float

class PricingSimulationResponse(BaseModel):
    route: str
    demand_scenario: str
    base_fare: float
    capacity: int
    results: List[PricingStrategyResult]
    recommended_strategy: str
