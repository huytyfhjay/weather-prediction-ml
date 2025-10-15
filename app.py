# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

st.set_page_config(page_title="Weather Prediction (ML)", layout="wide")

st.title("Weather Prediction Web App — ML (RandomForest)")
st.markdown("Upload a CSV with features like temperature, humidity, wind_speed, pressure, etc., and a target column (e.g., `next_day_temp` or `temperature`). The app will train a RandomForest regressor and let you make predictions.")

# Sidebar for instructions & upload
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Prepare CSV with numeric features and one target column (e.g., `next_day_temp`).  
2. Upload the CSV below.  
3. Select features and target.  
4. Train model and test.  
5. Use manual inputs to predict single-row results.  
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use built-in sample data (if you don't have CSV)")

@st.cache_data
def load_sample():
    # a small realistic-ish sample
    rng = np.random.RandomState(42)
    n = 500
    data = pd.DataFrame({
        "temp": rng.normal(25, 5, n),           # current day avg temp (°C)
        "humidity": rng.uniform(30, 90, n),    # %
        "wind_speed": rng.uniform(0, 12, n),   # m/s
        "pressure": rng.normal(1013, 7, n),    # hPa
        "precip": rng.exponential(0.5, n),     # mm
    })
    # target: next_day_temp (some dependence + noise)
    data["next_day_temp"] = data["temp"] * 0.6 + (25 - data["humidity"] * 0.02) + rng.normal(0, 1.8, n)
    return data

if use_sample and uploaded_file is None:
    df = load_sample()
    st.success("Loaded sample dataset (500 rows).")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded `{uploaded_file.name}` — {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        st.error("Could not read CSV. Make sure it's a valid CSV file.")
        st.stop()
else:
    st.info("Upload a CSV or check 'Use built-in sample data' to proceed.")
    st.stop()

st.subheader("Preview data")
st.dataframe(df.head(10))

# Let user choose target
all_columns = df.columns.tolist()
target_col = st.selectbox("Select target column (what to predict)", options=all_columns, index=len(all_columns)-1 if len(all_columns)>0 else 0)
feature_cols = st.multiselect("Select feature columns (inputs)", options=[c for c in all_columns if c != target_col],
                              default=[c for c in all_columns if c != target_col][:5])

if len(feature_cols) == 0:
    st.warning("Choose at least one feature column.")
    st.stop()

# Basic data cleaning: drop rows with NaNs in selected cols
df_model = df[feature_cols + [target_col]].dropna()
st.write(f"Using {df_model.shape[0]} rows after dropping missing values.")

# Train-test split settings
st.sidebar.subheader("Training settings")
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
n_estimators = st.sidebar.slider("RandomForest n_estimators", 10, 300, 100, 10)
max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 50, 0, 1)

X = df_model[feature_cols].values
y = df_model[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train button
if st.button("Train RandomForest"):
    st.info("Training model... hang on")
    if max_depth == 0:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    # predictions
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.success("Training complete.")
    st.subheader("Evaluation on test set")
    st.write(f"Mean Absolute Error: **{mae:.3f}**")
    st.write(f"RMSE: **{rmse:.3f}**")
    st.write(f"R² score: **{r2:.3f}**")

    # Feature importances
    try:
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.subheader("Feature importances")
        st.table(fi.to_frame("importance").round(4))
        # plot
        fig, ax = plt.subplots()
        sns.barplot(x=fi.values, y=fi.index, ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except Exception:
        pass

    # Save model
    save_path = st.text_input("Save model to (filename)", value="rf_weather_model.joblib")
    if st.button("Save model file"):
        joblib.dump({"model": model, "features": feature_cols}, save_path)
        st.success(f"Model saved to `{save_path}` in working directory.")

    # Show sample predictions vs true
    st.subheader("Sample predictions (first 10 rows of test set)")
    sample_df = pd.DataFrame(X_test, columns=feature_cols).head(10)
    sample_df[target_col + "_true"] = y_test[:10]
    sample_df[target_col + "_pred"] = y_pred[:10]
    st.dataframe(sample_df.round(3))

    # Allow single-row manual prediction
    st.subheader("Make a manual single prediction")
    manual_vals = {}
    cols = st.columns(len(feature_cols))
    for i, f in enumerate(feature_cols):
        # pick a reasonable default from data median
        default = float(np.nanmedian(df_model[f]))
        manual_vals[f] = cols[i].number_input(f"Input `{f}`", value=round(default, 3))
    if st.button("Predict (manual)"):
        input_arr = np.array([manual_vals[f] for f in feature_cols]).reshape(1, -1)
        pred = model.predict(input_arr)[0]
        st.write(f"Predicted **{target_col}** = **{pred:.3f}**")

else:
    st.info("Press 'Train RandomForest' to train model and see results.")

st.markdown("---")
st.caption("App created for exam/demo. If you need a packaged zip or Heroku/Streamlit Cloud deployment steps, ask and I'll add them.")
