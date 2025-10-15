# generate_sample_data.py
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
n = 500
data = pd.DataFrame({
    "temp": rng.normal(25, 5, n),
    "humidity": rng.uniform(30, 90, n),
    "wind_speed": rng.uniform(0, 12, n),
    "pressure": rng.normal(1013, 7, n),
    "precip": rng.exponential(0.5, n),
})
data["next_day_temp"] = data["temp"] * 0.6 + (25 - data["humidity"] * 0.02) + rng.normal(0, 1.8, n)
data.to_csv("sample_weather.csv", index=False)
print("Saved sample_weather.csv")
