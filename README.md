# FareSynth AI: Synthetic Airline Fare and Booking Data Generator

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://mahidharreddy06.github.io/FareSynth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

FareSynth AI is a synthetic airline fare and booking time-series generator designed for dynamic pricing model experimentation, revenue management stress-testing, and yield optimization. The system is implemented using **TimeGAN** (Time-series Generative Adversarial Networks) in PyTorch, with empirical statistical validation and a production-grade FastAPI microservice.

---

## Project Specifications

- **Core Frameworks**: Python 3.13, PyTorch 2.11, Pandas, FastAPI, Scikit-Learn, SciPy
- **Domain Scope**: Multivariate Airline Revenue Management and Dynamic Pricing
- **Sequential Features**: Days to departure, demand index, available seat inventory, passenger load factor, daily bookings, competitor fare, dynamic ticket fare
- **Validation**: Kolmogorov-Smirnov two-sample test, 1D Wasserstein distance, Frobenius correlation norm
- **Deployment**: REST API with interactive Swagger documentation and web dashboard

---

## Problem Statement and Objectives

Airlines require diverse, non-stationary booking and fare data to train and evaluate dynamic pricing policies. Historical datasets often lack extreme market conditions, such as sudden event-driven surges or off-peak volatility. Furthermore, deploying exploratory algorithms directly into live booking engines introduces commercial risk.

FareSynth AI addresses this challenge by generating realistic, multi-attribute temporal sequences that reproduce:
1. Cross-sectional relationships between competitor fares, seat scarcity, and price sensitivity.
2. Stepwise temporal transitions over a 30-day booking horizon.
3. Physical airline domain invariants, including monotonic seat consumption and bounded load factors.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                            FareSynth AI Architecture                              |
+-----------------------------------------------------------------------------------+
                                          |
  [ Historical Flight Simulation ]        |   [ Time-Series Preprocessing ]
  - Routes (JFK-LHR, LAX-ORD, etc.)       |   - Feature Selection (7 dimensions)
  - Booking window (30 to 1 days)         |   - MinMaxScaler normalization [0, 1]
  - Seasonal & Demand multipliers         |   - 3D Windowing: [N, 30, 7]
                                          v
+-----------------------------------------------------------------------------------+
|                        TimeGAN Generative Architecture                            |
|                                                                                   |
|    Phase 1: Autoencoder Pretraining (Embedder -> Recovery)                        |
|    Phase 2: Stepwise Supervisor Pretraining (Latent Temporal Dynamics)            |
|    Phase 3: Joint Adversarial Training (Generator vs Discriminator + Moments)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
  [ Synthetic Sequence Generation & Post-Processing ]
  - Latent noise sampling Z ~ N(0, 1) -> E -> H -> X_fake
  - Inverse scaling to physical airline units ($ / seats / load factor)
  - Domain constraint enforcement (non-negative, bounded load factor)
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-----------------------------+                           +-----------------------------+
| Comprehensive Validation    |                           | Dynamic Pricing Sandbox     |
| - Data Integrity Checks     |                           | - Rule-Based Yield Manager  |
| - Kolmogorov-Smirnov Test   |                           | - Competitor Matching       |
| - Wasserstein Distance      |                           | - Adaptive Q-Learning RL    |
| - Correlation Frobenius Norm|                           | - Revenue & RASM Benchmark  |
| - 2D PCA & Latent Overlap   |                           +-----------------------------+
+-----------------------------+                                         |
                                                                        v
                                                          +-----------------------------+
                                                          | FastAPI Microservice Engine |
                                                          | - GET /health               |
                                                          | - GET /routes               |
                                                          | - POST /generate            |
                                                          | - POST /pricing/simulate    |
                                                          | - Interactive Swagger /docs |
                                                          +-----------------------------+
```

---

## Project Milestones and Curriculum Mapping

| Milestone | Bloom's Taxonomy | Weight | Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1: Fare Data Preparation & TimeGAN Setup** | *Apply* | 15% | `data/historical_airline_fares.csv`, `src/preprocessor.py` |
| **Week 2: Synthetic Airline Fare Generation** | *Create* | 25% | `src/timegan.py`, `src/trainer.py`, `data/synthetic_airline_fares.csv` |
| **Week 3: Fare Validation & Demand Analysis** | *Evaluate & Analyze* | 35% | `reports/synthetic_fare_validation_report.md`, `reports/figures/` |
| **Week 4: FastAPI & Dynamic Pricing Testing** | *Create & Evaluate* | 25% | `app/main.py`, `src/dynamic_pricing.py`, Interactive API |

---

## Task Breakdown and Implementation Status

- [x] **Task 1**: Directory structure and dependency configuration.
- [x] **Task 2**: Airline revenue management domain schema (routes, capacities, seasons).
- [x] **Task 3**: Historical airline fare simulation with empirical elasticity distributions.
- [x] **Task 4**: Preprocessing pipeline with normalization and 3D sliding sequence windowing.
- [x] **Task 5**: PyTorch TimeGAN architecture (Embedder, Recovery, Generator, Supervisor, Discriminator).
- [x] **Task 6**: Autoencoder pretraining with reconstruction loss.
- [x] **Task 7**: Supervisor pretraining on hidden state sequence transitions.
- [x] **Task 8**: Joint adversarial min-max training with moment-matching regularizer.
- [x] **Task 9**: Model checkpointing and parameter serialization.
- [x] **Task 10**: Synthetic airline fare generation and CSV export.
- [x] **Task 11**: Data integrity validation (completeness, non-negativity, capacity constraints).
- [x] **Task 12**: Two-sample Kolmogorov-Smirnov test and Wasserstein distance evaluation.
- [x] **Task 13**: Correlation matrix Frobenius norm computation and cross-temporal analysis.
- [x] **Task 14**: 2D PCA latent trajectory projection and marginal distribution visualization.
- [x] **Task 15**: Multi-route, seasonal, and surge demand pattern analysis.
- [x] **Task 16**: Dynamic pricing simulation engine (Baseline, Yield Manager, Competitor Follower, Q-Learner).
- [x] **Task 17**: Dynamic pricing stress-testing and revenue comparison across surge scenarios.
- [x] **Task 18**: FastAPI request and response validation schemas.
- [x] **Task 19**: REST endpoints (`/health`, `/routes`, `/generate`, `/pricing/simulate`).
- [x] **Task 20**: Automated pytest test suite (12/12 passing).
- [x] **Task 21**: Project documentation and end-to-end execution pipeline.

---

## Getting Started and Execution

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/mahidharreddy06/FareSynth.git
cd FareSynth
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

To train the TimeGAN model, export synthetic records, run statistical validation, and simulate pricing models:

```bash
python run_pipeline.py
```

### 3. Launch the Web Application and API

To start the server with the web interface and REST API:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Access points:
- Web Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4. Run Automated Test Suite

```bash
python -m pytest tests/ -v
```

---

## Empirical Evaluation and Results

### Validation Integrity

- **Missing Values**: 0 nulls across real and synthetic datasets.
- **Physical Bounds**: Fares strictly positive; load factors bounded within [0.0, 1.0].
- **Correlation Fidelity**: Low Frobenius norm divergence between correlation matrices (3.568).

### Dynamic Pricing Testing Insights

When evaluated across synthetic demand conditions:
1. **Rule-Based Yield Management** achieves 18% to 24% higher revenue than static baseline pricing by adjusting fares dynamically as seat inventory tightens.
2. **Adaptive Q-Learning** balances load factor with revenue optimization, preventing early seat exhaustion during surge demand while maintaining load factor during off-peak periods.

---

## Deliverables and Repository Contents

- Historical Fare Dataset: `data/historical_airline_fares.csv`
- Synthetic Fare Dataset: `data/synthetic_airline_fares.csv`
- Trained TimeGAN Weights: `models/timegan_fare_model.pt`
- Scaler Parameters: `models/scaler_params.json`
- Validation Report: `reports/synthetic_fare_validation_report.md`
- Visual Analytics Figures: `reports/figures/`
- FastAPI Application: `app/main.py`
- Frontend Dashboard: `frontend/index.html`
