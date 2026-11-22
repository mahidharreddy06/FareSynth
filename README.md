# FareSynth AI: Synthetic Airline Fare Data Generator

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%202.11-red.svg)](https://pytorch.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

A state-of-the-art synthetic airline fare and booking time-series data generator built for **dynamic-pricing model experimentation and yield management testing**. Powered by **TimeGAN** (Time-series Generative Adversarial Networks), PyTorch, Pandas, and FastAPI.

---

## 📋 Project Specifications
- **Core Technology**: Python 3.13 | PyTorch 2.11 (TimeGAN) | Pandas | FastAPI | Scikit-Learn
- **Domain Scope**: Multivariate Airline Revenue Management & Dynamic Pricing
- **Validation**: Kolmogorov-Smirnov Test, Wasserstein Distance, Correlation Matrix Frobenius Fidelity
- **Deployment**: Microservice REST API & Interactive Flight Operations Dashboard

---

## 🎯 Problem Statement & Objectives
Airlines rely on complex dynamic pricing algorithms to optimize revenue across thousands of flight departures daily. However:
1. **Data Scarcity & Privacy Constraints**: Historical datasets often lack extreme demand scenarios (e.g. sudden sports events, viral conferences, extreme weather surges).
2. **Safe Experimentation**: Testing exploratory reinforcement learning (RL) pricing agents directly on production flight inventory risks severe revenue loss and seat spoilage.

**FareSynth AI** generates realistic, multivariate time-series sequences of airline fares, seat availability, passenger load factors, booking velocity, and competitor prices across multiple routes, seasons, and demand scenarios.

---

## 🏗️ System Architecture & Workflow

```
+-----------------------------------------------------------------------------------+
|                            FareSynth AI Architecture                              |
+-----------------------------------------------------------------------------------+
                                          |
  [ Historical Flight Simulation ]        |   [ Time-Series Preprocessing ]
  - Routes (JFK-LHR, LAX-ORD, etc.)       |   - Feature Selection (7 dimensions)
  - Booking window (30 -> 1 days)         |   - MinMaxScaler normalization [0, 1]
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
  - Inverse scaling back to physical airline units ($ / seats / load factor)
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

## 📊 Bloom's Taxonomy & Milestone Mapping

| Milestone / Module | Bloom's Level | Evaluation Weight | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1: Fare Data Prep & TimeGAN Setup** | *Apply* | 15% | `data/historical_airline_fares.csv`, `src/preprocessor.py` |
| **Week 2: Synthetic Airline Fare Generation** | *Create* | 25% | `src/timegan.py`, `src/trainer.py`, `data/synthetic_airline_fares.csv` |
| **Week 3: Fare Validation & Demand Analysis** | *Evaluate & Analyze* | 35% | `reports/synthetic_fare_validation_report.md`, `reports/figures/` |
| **Week 4: FastAPI & Dynamic Pricing Testing** | *Create & Evaluate* | 25% | `app/main.py`, `src/dynamic_pricing.py`, Interactive API |

---

## 📋 The 21 Project Tasks Breakdown

- [x] **Task 1**: Environment setup, folder architecture (`data/`, `src/`, `app/`, `models/`, `reports/`, `tests/`).
- [x] **Task 2**: Airline domain schema design (Routes, Cabin capacities, Seasons, Demand indices).
- [x] **Task 3**: Historical airline fare simulation script with authentic macroeconomic distributions (`src/data_generator.py`).
- [x] **Task 4**: Preprocessing pipeline with MinMaxScaler and 3D sliding sequence windowing (`src/preprocessor.py`).
- [x] **Task 5**: PyTorch TimeGAN architecture implementation (Embedder, Recovery, Generator, Supervisor, Discriminator) (`src/timegan.py`).
- [x] **Task 6**: Autoencoder pretraining phase with reconstruction loss ($\mathcal{L}_R$).
- [x] **Task 7**: Supervisor pretraining phase on hidden sequence transitions ($\mathcal{L}_S$).
- [x] **Task 8**: Joint adversarial training with moment matching and dual discriminators.
- [x] **Task 9**: Model checkpointing and parameter serialization (`models/timegan_fare_model.pt`).
- [x] **Task 10**: Synthetic airline fare generation and export (`data/synthetic_airline_fares.csv`).
- [x] **Task 11**: Data integrity and business constraint validation (non-negativity, capacity checks).
- [x] **Task 12**: Two-sample Kolmogorov-Smirnov test and Wasserstein distance evaluation.
- [x] **Task 13**: Correlation matrix Frobenius norm computation and cross-temporal dependence check.
- [x] **Task 14**: 2D PCA latent trajectory projection and marginal distribution visualization.
- [x] **Task 15**: Multi-route, seasonal, and surge demand pattern analysis (`src/analysis.py`).
- [x] **Task 16**: Dynamic pricing simulation engine (Rule-Based Yield Manager, Competitor Follower, Q-Learner).
- [x] **Task 17**: Dynamic pricing stress-testing and revenue comparison across surge scenarios.
- [x] **Task 18**: FastAPI Pydantic request/response schemas (`app/schemas.py`).
- [x] **Task 19**: FastAPI application endpoints (`GET /health`, `GET /routes`, `POST /generate`, `POST /pricing/simulate`).
- [x] **Task 20**: Comprehensive automated pytest test suite (`tests/`).
- [x] **Task 21**: Project documentation, execution pipeline, and deliverable report packaging.

---

## 🚀 Quickstart & Usage Instructions

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Generate historical data, train TimeGAN, export synthetic data, run validation, and execute dynamic pricing experiments in one command:
```bash
python run_pipeline.py
```

### 3. Launch the FastAPI Microservice
Start the interactive API server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to:
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 📈 Evaluation & Results Summary

### Validation Integrity
- **Missing Values**: 0 nulls across real and synthetic datasets.
- **Physical Bounds**: 100% positive fares, load factors strictly bounded in $[0.0, 1.0]$.
- **Correlation Fidelity**: Low Frobenius norm divergence between correlation matrices.

### Dynamic Pricing Testing Insights
When tested against synthetic extreme demand scenarios:
1. **Rule-Based Yield Management** achieves 18-24% higher revenue than static baseline pricing by capturing last-minute business traveler surplus.
2. **Adaptive Q-Learning** balances load factor with revenue optimization, preventing premature seat sell-out during high demand while discounting early during low demand.

---

## 📜 Deliverables Included
- Clean historical airline fare dataset: `data/historical_airline_fares.csv`
- Generated synthetic airline fare dataset: `data/synthetic_airline_fares.csv`
- Checkpointed trained TimeGAN model: `models/timegan_fare_model.pt`
- Comprehensive validation report: `reports/synthetic_fare_validation_report.md`
- Visual analytics and validation figures: `reports/figures/`
- Production FastAPI service: `app/main.py`
