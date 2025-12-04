from pathlib import Path

import joblib
import pandas as pd

from .data_prep import load_clean_daily

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def load_model():
    """
    Load the trained RandomForest model.
    """
    model_path = MODELS_DIR / "onion_maharashtra_rf.pkl"
    model = joblib.load(model_path)
    return model


def make_forecast(horizon_days: int = 7) -> pd.DataFrame:
    """
    Iterative multi-step forecast for `horizon_days` into the future.

    Uses the trained RF model and simulates future days by
    updating lag and rolling features with predicted values.
    """

    model = load_model()
    daily = load_clean_daily()

    # Ensure sorted chronologically
    daily = daily.sort_values("Date")

    # We'll keep a Series of historical + predicted prices
    prices = daily.set_index("Date")["Avg_Modal_Price"].copy()
    prices = prices.sort_index()

    last_date = prices.index.max()

    forecast_rows = []

    for i in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=i)

        # History up to this point (includes previous predictions)
        history = prices.sort_index()

        # Lags
        lag_1 = history.iloc[-1]
        lag_3 = history.iloc[-3] if len(history) >= 3 else lag_1
        lag_7 = history.iloc[-7] if len(history) >= 7 else lag_3

        # Rolling windows
        last_7 = history.iloc[-7:] if len(history) >= 7 else history
        roll_mean_7 = last_7.mean()
        roll_std_7 = last_7.std() if len(last_7) > 1 else 0.0

        last_14 = history.iloc[-14:] if len(history) >= 14 else history
        roll_mean_14 = last_14.mean()

        # Time features
        dow = future_date.weekday()
        month = future_date.month
        weekofyear = future_date.isocalendar().week

        feature_row = pd.DataFrame(
            [
                {
                    "day_of_week": dow,
                    "month": month,
                    "weekofyear": int(weekofyear),
                    "lag_1": lag_1,
                    "lag_3": lag_3,
                    "lag_7": lag_7,
                    "roll_mean_7": roll_mean_7,
                    "roll_std_7": roll_std_7,
                    "roll_mean_14": roll_mean_14,
                }
            ]
        )

        pred_price = float(model.predict(feature_row)[0])

        forecast_rows.append(
            {
                "Date": future_date,
                "Predicted_Price": pred_price,
            }
        )

        # Append this prediction to history so next step can use it
        prices.loc[future_date] = pred_price

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    return forecast_df