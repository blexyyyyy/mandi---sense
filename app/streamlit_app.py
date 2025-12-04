import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable when running `streamlit run app/streamlit_app.py`
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_prep import load_clean_daily
from src.forecast_service import make_forecast

st.set_page_config(
    page_title="Mandi-Sense: Onion Price Forecast",
    layout="wide",
)

st.title("🧅 Mandi-Sense: Onion Price Forecast – Maharashtra")

st.markdown(
    """
    This tool uses historical **mandi onion prices in Maharashtra**  
    and a **RandomForest regression model** to forecast short-term prices.
    """
)

# Load historical data
daily = load_clean_daily()
daily = daily.sort_values("Date")

st.sidebar.header("Forecast controls")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=14, value=7)

st.sidebar.markdown(
    """
    **How to read the chart:**
    - Solid line = historical average modal price (₹ per quintal)
    - Dotted/extended line = model forecast
    """
)

# Make forecast
forecast_df = make_forecast(horizon_days=horizon)

# Combine last N days of history + forecast for plotting
history_window_days = 90
history_tail = daily.tail(history_window_days).copy()

history_tail = history_tail[["Date", "Avg_Modal_Price"]]
history_tail = history_tail.rename(columns={"Avg_Modal_Price": "Historical_Price"})

plot_df = pd.merge(
    history_tail,
    forecast_df,
    how="outer",
    on="Date",
)

plot_df = plot_df.set_index("Date")

st.subheader("Price history and forecast")

st.line_chart(plot_df)

st.subheader("Forecast details")
st.dataframe(forecast_df.style.format({"Predicted_Price": "{:.2f}"}))

st.markdown(
    """
    **Note:** This is a demo model.  
    Real-world deployment would:
    - retrain regularly,
    - include more features (weather, arrivals, policy events),
    - and monitor forecast error over time.
    """
)