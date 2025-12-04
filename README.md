# Mandi-Sense: Onion Price Forecasting for Maharashtra Mandis

Mandi-Sense is a small end-to-end ML project that forecasts **short-term onion prices** for **Maharashtra mandis** using historical government mandi data.

The goal is to mimic the kind of **decision-support tooling** that commodity platforms (like NeML / agri-exchanges / e-mandis) would use to:

- Anticipate short-term price movements  
- Help traders / buyers / policymakers plan better  
- Demonstrate an ML pipeline on real Indian agri-market data

---

## 1. Problem Statement

Onion prices in India are volatile and politically sensitive.  
Maharashtra is home to some of the largest onion markets (e.g. Lasalgaon, Nashik), and prices can swing sharply within days.

**Objective:**  
Given historical daily mandi prices, build a model that can:

- Forecast **next 7–14 days** average modal onion price (₹) for Maharashtra  
- Beat a naive baseline (tomorrow = today’s price)  
- Expose the forecasts through a simple web UI

This is NOT about perfect prediction. It’s about building a **clean, explainable ML pipeline** on noisy real-world data.

---

## 2. Data

The dataset is derived from **Indian government mandi price data (AGMARKNET-style)** and contains:

- `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`
- `Min_Price`, `Max_Price`, `Modal_Price`
- `Date` (daily prices)

For this project, we:

- Filtered to **Onion** in **Maharashtra**  
- Used data roughly from **July 2023 to June 2025**  
- Aggregated to a **state-wide daily time series**:
  - `Avg_Modal_Price` (mean of modal prices across markets)
  - `Min_Price`, `Max_Price`
  - `Num_Markets` (number of markets reporting that day)

> Note: Raw data files are not committed to GitHub (to keep repo light).  
> Only processed/feature data or instructions may be included.

---

## 3. Approach

### 3.1. Aggregation & EDA

1. **Filter**: `Commodity == "Onion"` and `State == "Maharashtra"`
2. **Clean**:
   - Parsed `Date` to `datetime`
   - Converted price columns to numeric
   - Dropped rows with missing `Modal_Price`
   - Clipped extreme outliers at [1st, 99th] percentile
3. **Aggregate** to 1 row per `Date`:
   - `Avg_Modal_Price` = mean modal price across all markets that day
   - `Num_Markets` = number of markets reporting that day
4. **EDA**:
   - Time-series plot of `Avg_Modal_Price`
   - Monthly average price plot to see seasonality

### 3.2. Features

On the daily aggregated series, we engineered:

- **Time-based features**
  - `day_of_week` (0–6)
  - `month` (1–12)
  - `weekofyear` (1–53)
- **Lag features**
  - `lag_1`  – price 1 day ago  
  - `lag_3`  – price 3 days ago  
  - `lag_7`  – price 7 days ago  
- **Rolling window features**
  - `roll_mean_7`  – 7-day rolling mean  
  - `roll_std_7`   – 7-day rolling std  
  - `roll_mean_14` – 14-day rolling mean  

Rows where lags/rolling windows were undefined (start of series) were dropped, producing a clean ML-ready dataset.

### 3.3. Train / Validation Split

Time-based split (no shuffling):

- **Train:** ~80% of days  
- **Validation:** last ~20% of days  
- Roughly:
  - Train: `2023-07-07` → `2025-02-10`  
  - Valid: `2025-02-11` → `2025-06-11`

---

## 4. Models & Performance

### 4.1. Baseline (Naive)

**Naive baseline**: predict that today’s price = yesterday’s price (`lag_1`).

Metrics on validation set:

- `MAE`  = *fill from notebook*  
- `RMSE` = *fill from notebook*  
- `MAPE` = *fill from notebook* %

*(You can fill these directly from your `04_modeling.ipynb` output.)*

### 4.2. RandomForest Regressor

Model:

- `RandomForestRegressor`
  - `n_estimators = 400`
  - `max_depth = 12`
  - `random_state = 42`
  - `n_jobs = -1`

Metrics on validation set:

- `MAE`  ≈ **134.21**  
- `RMSE` ≈ **168.03**  
- `MAPE` ≈ **11.21%**

The RandomForest model improves over the naive baseline and captures some of the short-term structure in onion prices, despite noise and sudden spikes.

---

## 5. Forecasting Logic

The forecasting function performs **iterative multi-step prediction**:

1. Start with the latest historical aggregated daily series.
2. For each future day (1..N):
   - Compute lag features (`lag_1`, `lag_3`, `lag_7`) from the current history (including previous predictions).
   - Compute rolling features (`roll_mean_7`, `roll_std_7`, `roll_mean_14`) from the current history.
   - Add time features for the future date (day-of-week, month, week-of-year).
   - Use the trained RandomForest to predict the **next-day price**.
   - Append the prediction to the history and move to the next day.
3. Return a small DataFrame of future dates and predicted prices.

This is implemented in:

- `src/data_prep.py`
- `src/forecast_service.py`

---

## 6. Streamlit App

The app lives in `app/streamlit_app.py`.

Features:

- Shows **last 90 days** of historical average onion price in Maharashtra
- Lets the user pick **forecast horizon** (1–14 days)
- Plots a combined chart:
  - Historical prices
  - Forecasted prices
- Displays a **forecast table** with predicted prices

### Running the app

From project root:

```bash
# (Optional) activate virtualenv
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app/streamlit_app.py
