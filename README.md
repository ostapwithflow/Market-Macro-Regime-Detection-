# Macro Regime Engine — Slow Tower (Layer 1)

**Macro-structural market regime classification** using Hidden Markov Models and Federal Reserve liquidity data.

Classifies the market into three structural regimes — **CALM**, **TRANSITION**, **STRESS** — based on federal liquidity flows and the yield curve, with an anomaly detection gatekeeper for Black Swan protection.

> This is the **macro layer only** (Slow Tower) extracted from the full Regime Engine v3.0 "Quant Village" two-tower system.

---

## Architecture

```
FRED API Data (8 series)
    │
    ▼
┌─────────────────────────┐
│  Fred Client             │  WALCL, RRPONTSYD, WTREGEN, DGS10,
│  (data/fred_client.py)   │  DGS3MO, DGS2, BAMLH0A0HYM2, T5YIE
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Macro Features          │  Net Liquidity = (WALCL/1000) - RRP - TGA
│  (data/macro_features.py)│  Yield Spread = DGS10 - DGS3MO
│                          │  → EWMA Z-scores (halflife=60d)
│  X_t = [L_t, S_t]       │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌──────────┐  ┌──────────────────┐
│ OOD Gate │  │ Gaussian HMM     │
│ (MCD)    │  │ (3 states, full  │
│ χ² 99.9% │  │  cov, 7 restarts)│
└────┬─────┘  └────────┬─────────┘
     │                 │
     │    ┌────────────┤
     ▼    ▼            ▼
  OOD Override    State Probabilities
  → STRESS        → Transition Matrix
                  → Tomorrow Forecast
                        │
                        ▼
              ┌─────────────────┐
              │  CALM            │
              │  TRANSITION      │
              │  STRESS          │
              └─────────────────┘
```

---

## Emission Vector

| Feature | Formula | Description |
|---------|---------|-------------|
| **L_t** | EWMA Z-score of Net Liquidity | Federal liquidity impulse |
| **S_t** | EWMA Z-score of Yield Spread (10Y−3M) | Yield curve shape signal |

**Net Liquidity** = (Fed Balance Sheet / 1000) − Reverse Repo − TGA

---

## Models

### Gaussian HMM (3 States)
- Full covariance matrix
- 7 random restarts (best log-likelihood wins)
- Semantic alignment: 0=CALM, 1=TRANSITION, 2=STRESS (by sum of feature means)
- Tomorrow forecast: `P(t+1) = Transition Matrix' × P(t)`

### OOD Gatekeeper
- **MCD (Minimum Covariance Determinant)** — robust covariance estimation
- **Mahalanobis distance** — χ² threshold at 99.9 percentile
- **Cluster OOD** — additional Mahalanobis to HMM centroids
- Any OOD detection → automatic override to **STRESS**

---

## Project Structure

```
macro-regime-engine/
├── main.py                    # Live dashboard — entry point
├── config.py                  # API keys from env vars + FRED tickers
├── requirements.txt           # Python dependencies
├── RUN_ENGINE.bat             # One-click Windows launcher
├── .env.example               # Template for API keys
├── .gitignore
├── data/
│   ├── fred_client.py         # FredClient — fetches 8 FRED series
│   ├── macro_features.py      # Net Liquidity, Yield Spread → EWMA Z-scores
│   └── economic_calendar.py   # FOMC/CPI/NFP event dates
├── models/
│   ├── hmm_core.py            # RegimeEngineHMM — fit, align, predict
│   └── ood_gatekeeper.py      # OODGatekeeper — MCD + Mahalanobis
└── diagnostics/
    ├── collapse_detector.py   # Centroid collapse detection
    └── separability.py        # Cohen's d feature separability
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/macro-regime-engine.git
cd macro-regime-engine
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your FRED API key
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
```

Or set environment variables directly:
```bash
export FRED_API_KEY=your_key_here
```

### 3. Run

```bash
python main.py
```

Or on Windows: double-click `RUN_ENGINE.bat`

---

## Output Example

```
================================================================================
              MACRO REGIME ENGINE — SLOW TOWER (Layer 1)
================================================================================
 Останнє оновлення:  2026-05-30 (t)
 Прогноз на сесію:   2026-05-31 (t+1)
 Аномалія (OOD):     [OK] NOMINAL (Normal)
 Макро-Подія:        Ні
--------------------------------------------------------------------------------
 [MACRO REGIME]  Структурний режим:     CALM
                 Ймовірності станів:    CALM: 87.3% | TRANSITION: 10.2% | STRESS: 2.5%
                 Стабільність режиму:   96.8%
                 Впевненість (Cert):    91.6%
--------------------------------------------------------------------------------
 [TRANSITION MATRIX]
                     CALM     TRANS    STRESS
      CALM  →    96.8%     2.9%     0.3%
     TRANS  →    12.1%    82.4%     5.5%
    STRESS  →     1.2%    15.3%    83.5%
================================================================================
```

---

## FRED Data Series

| Ticker | Description |
|--------|-------------|
| `WALCL` | Fed Total Assets (Millions $) |
| `RRPONTSYD` | Reverse Repo (Billions $) |
| `WTREGEN` | Treasury General Account (Billions $) |
| `DGS10` | 10-Year Treasury Yield |
| `DGS3MO` | 3-Month Treasury Yield |
| `DGS2` | 2-Year Treasury Yield |
| `BAMLH0A0HYM2` | High Yield Spread |
| `T5YIE` | 5-Year Breakeven Inflation |

---


