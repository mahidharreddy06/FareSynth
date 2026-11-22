import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

class AirlinePatternAnalyzer:
    """
    Performs analytical evaluation on synthetic vs historical airline data
    focusing on routes, seasons, demand scenarios, and pricing dynamics.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def route_pricing_summary(self) -> pd.DataFrame:
        """Computes fare statistics, average capacity, and load factor by route."""
        grouped = self.df.groupby("route").agg(
            avg_fare=("fare_price", "mean"),
            min_fare=("fare_price", "min"),
            max_fare=("fare_price", "max"),
            avg_load_factor=("load_factor", "mean"),
            avg_daily_bookings=("daily_bookings", "mean")
        ).round(2)
        return grouped

    def seasonal_pricing_summary(self) -> pd.DataFrame:
        """Computes pricing metrics across seasons."""
        grouped = self.df.groupby("season").agg(
            avg_fare=("fare_price", "mean"),
            avg_demand_index=("demand_index", "mean"),
            avg_load_factor=("load_factor", "mean"),
            avg_bookings=("daily_bookings", "mean")
        ).round(2)
        return grouped

    def demand_scenario_summary(self) -> pd.DataFrame:
        """Computes pricing metrics across demand scenarios (Low, Normal, High, Surge)."""
        grouped = self.df.groupby("demand_scenario").agg(
            avg_fare=("fare_price", "mean"),
            avg_demand_index=("demand_index", "mean"),
            avg_load_factor=("load_factor", "mean"),
            total_bookings=("daily_bookings", "sum")
        ).round(2)
        return grouped

    def generate_analytical_visualizations(self, output_dir: Path):
        """Generates visual analytics charts for demand, route, and seasonal patterns."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Route Comparison Bar Plot
        route_stats = self.route_pricing_summary().reset_index()
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        bars = ax1.bar(route_stats["route"], route_stats["avg_fare"], color="#2471a3", alpha=0.8, width=0.45, label="Avg Fare ($)")
        line = ax2.plot(route_stats["route"], route_stats["avg_load_factor"] * 100, color="#c0392b", marker="o", lw=2.5, label="Avg Load Factor (%)")

        ax1.set_xlabel("Flight Route", fontweight="bold")
        ax1.set_ylabel("Average Fare ($)", color="#2471a3", fontweight="bold")
        ax2.set_ylabel("Load Factor (%)", color="#c0392b", fontweight="bold")
        ax1.set_title("Route Pricing & Passenger Load Factor Distribution", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_dir / "route_pricing_analysis.png", dpi=200)
        plt.close()

        # 2. Demand Scenario Pricing Surge Curves
        plt.figure(figsize=(10, 6))
        for scen in ["Low", "Normal", "High", "Surge"]:
            subset = self.df[self.df["demand_scenario"] == scen]
            if not subset.empty:
                curve = subset.groupby("days_to_departure")["fare_price"].mean().reset_index()
                plt.plot(curve["days_to_departure"], curve["fare_price"], marker="o", label=f"{scen} Demand", lw=2)

        plt.gca().invert_xaxis()
        plt.title("Fare Escalation Across 30-Day Window by Demand Scenario", fontsize=12, fontweight="bold")
        plt.xlabel("Days to Departure (30 -> 1)", fontweight="bold")
        plt.ylabel("Mean Fare ($)", fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / "demand_scenario_curves.png", dpi=200)
        plt.close()

        # 3. Seasonal Booking Trajectories
        plt.figure(figsize=(10, 6))
        for season in ["Spring", "Summer", "Fall", "Winter"]:
            subset = self.df[self.df["season"] == season]
            if not subset.empty:
                curve = subset.groupby("days_to_departure")["load_factor"].mean().reset_index()
                plt.plot(curve["days_to_departure"], curve["load_factor"] * 100, label=season, lw=2.2)

        plt.gca().invert_xaxis()
        plt.title("Passenger Load Factor Build-up Across Seasons", fontsize=12, fontweight="bold")
        plt.xlabel("Days to Departure (30 -> 1)", fontweight="bold")
        plt.ylabel("Cumulative Load Factor (%)", fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / "seasonal_load_factors.png", dpi=200)
        plt.close()

        print(f"Analytical charts saved to: {output_dir}")
