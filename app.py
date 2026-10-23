import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="TrafficWise | Traffic Study", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1400px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1.1rem;}
.hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0; color: white;}
.hero p {margin: 0; color: #e0f2fe; font-size: 1rem;}
.section-label {color:#176b87; font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; margin-bottom:.2rem;}
div[data-testid="stMetric"] {background:#f4f8fb; border:1px solid #dce7ef; padding: .8rem 1rem; border-radius: 12px;}
.small-note {font-size:.88rem; color:#526575;}
</style>
<div class="hero"><h1>🚦 TrafficWise</h1><p>Urban Traffic Congestion Prediction and Area-Level Analysis</p></div>
""", unsafe_allow_html=True)
st.write("Explore historical congestion patterns, estimate the next recorded value for a selected road, and evaluate a simple machine-learning model against a baseline.")

with st.sidebar:
    st.header("Project controls")
    st.write("Upload the traffic dataset to run the analysis.")
    f = st.file_uploader("Traffic data (CSV)", type=["csv"])
    st.divider()
    st.caption("TrafficWise • Student research project")
    st.caption("Uses the uploaded dataset; no live traffic feed is connected.")

if f is None:
    st.info("To begin, upload **Bengaluru_Traffic_Data_Easy_To_Find.csv** using the panel on the left.")
    st.markdown("#### What this dashboard contains")
    a,b,c = st.columns(3)
    a.markdown("**01 · Road history**\n\nReview recorded congestion and a next-observation estimate.")
    b.markdown("**02 · Model experiment**\n\nCompare Linear Regression with a previous-value baseline using MAE and RMSE.")
    c.markdown("**03 · Area comparison**\n\nIdentify areas with higher average recorded congestion.")
    st.stop()

try:
    raw = pd.read_csv(f)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()
required = ["Date", "Area Name", "Road/Intersection Name", "Congestion Level"]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error("The uploaded file is missing required columns: " + ", ".join(missing))
    st.stop()

raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
raw["Congestion Level"] = pd.to_numeric(raw["Congestion Level"], errors="coerce")
raw = raw.dropna(subset=["Date", "Congestion Level", "Road/Intersection Name", "Area Name"]).copy()
if raw.empty:
    st.error("No usable rows were found. Check the date and congestion columns in your CSV.")
    st.stop()
# The dataset repeats source records for connected-route information; keep one copy per source ID.
model_df = raw.drop_duplicates("Source_Record_ID") if "Source_Record_ID" in raw.columns else raw.drop_duplicates(["Date", "Road/Intersection Name"])
model_df = model_df.copy()
model_df["Congestion Level"] = model_df["Congestion Level"].clip(0, 100)
model_df["Road/Intersection Name"] = model_df["Road/Intersection Name"].astype(str)
model_df["Area Name"] = model_df["Area Name"].astype(str)
model_df = model_df.sort_values(["Road/Intersection Name", "Date"]).copy()
for i in [1, 2, 3]:
    model_df[f"lag_{i}"] = model_df.groupby("Road/Intersection Name")["Congestion Level"].shift(i)
model_df["day"] = model_df["Date"].dt.dayofweek
model_df["month"] = model_df["Date"].dt.month
m = model_df.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()
if len(m) < 20:
    st.error("There are not enough records to build the historical features for this experiment.")
    st.stop()

unique_dates = np.sort(m["Date"].dt.normalize().unique())
if len(unique_dates) < 2:
    st.error("At least two distinct dates are needed for a chronological train/test split.")
    st.stop()
cut_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.8)))
cutoff = pd.Timestamp(unique_dates[cut_index])
train = m[m["Date"].dt.normalize() < cutoff].copy()
test = m[m["Date"].dt.normalize() >= cutoff].copy()
if train.empty or test.empty:
    st.error("Could not create non-empty training and testing sets from these dates.")
    st.stop()
feature_cols = ["lag_1", "lag_2", "lag_3", "day", "month", "Road/Intersection Name"]
X_train = pd.get_dummies(train[feature_cols], columns=["Road/Intersection Name"])
X_test = pd.get_dummies(test[feature_cols], columns=["Road/Intersection Name"]).reindex(columns=X_train.columns, fill_value=0)
model = LinearRegression().fit(X_train, train["Congestion Level"])
pred = np.clip(model.predict(X_test), 0, 100)
actual = test["Congestion Level"].to_numpy()
baseline = np.clip(test["lag_1"].to_numpy(), 0, 100)
model_mae = mean_absolute_error(actual, pred)
model_rmse = float(np.sqrt(mean_squared_error(actual, pred)))
base_mae = mean_absolute_error(actual, baseline)
base_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))
improvement = ((base_mae - model_mae) / base_mae * 100) if base_mae > 0 else np.nan

st.markdown('<div class="section-label">Dataset overview</div>', unsafe_allow_html=True)
k1,k2,k3,k4 = st.columns(4)
k1.metric("Unique records", f"{len(model_df):,}")
k2.metric("Areas", f"{model_df['Area Name'].nunique():,}")
k3.metric("Roads / intersections", f"{model_df['Road/Intersection Name'].nunique():,}")
k4.metric("Date coverage", f"{model_df['Date'].min():%d %b %Y} – {model_df['Date'].max():%d %b %Y}")
st.caption("Repeated connected-route rows are deduplicated when Source_Record_ID is available. Congestion values are bounded to 0–100 for display and modelling.")

st.divider()
st.markdown('<div class="section-label">01 · Road-level analysis</div>', unsafe_allow_html=True)
st.subheader("Recorded history and next-observation estimate")
roads = sorted(model_df["Road/Intersection Name"].unique())
road = st.selectbox("Select a road or intersection", roads)
rd = model_df[model_df["Road/Intersection Name"] == road].sort_values("Date").copy()
left,right = st.columns([2,1])
with left:
    fig = px.line(rd.tail(120), x="Date", y="Congestion Level", title="Historical congestion", range_y=[0,100])
    fig.update_traces(line=dict(color="#168aad", width=2.5))
    fig.update_layout(margin=dict(l=10,r=10,t=55,b=10), yaxis_title="Congestion (0–100)", xaxis_title="Recorded date")
    st.plotly_chart(fig, use_container_width=True)
with right:
    latest = float(rd["Congestion Level"].iloc[-1])
    st.metric("Latest recorded congestion", f"{latest:.1f} / 100")
    if latest < 35: level, colour = "Lower", "#2a9d8f"
    elif latest < 70: level, colour = "Moderate", "#e9c46a"
    else: level, colour = "Higher", "#e76f51"
    st.markdown(f"<div style='padding:.75rem 1rem;border-radius:10px;background:{colour}22;border-left:5px solid {colour};margin:.5rem 0 1rem 0'><b>{level} congestion band</b></div>", unsafe_allow_html=True)
    if len(rd) >= 3:
        vals = rd["Congestion Level"].tail(3).tolist()
        last_date = rd["Date"].max()
        next_date = last_date + pd.Timedelta(days=1)
        next_row = pd.DataFrame([{"lag_1":vals[-1],"lag_2":vals[-2],"lag_3":vals[-3],"day":next_date.dayofweek,"month":next_date.month,"Road/Intersection Name":road}])
        X_next = pd.get_dummies(next_row[feature_cols], columns=["Road/Intersection Name"]).reindex(columns=X_train.columns, fill_value=0)
        next_value = float(np.clip(model.predict(X_next)[0],0,100))
        st.metric("Estimated next observation", f"{next_value:.1f} / 100")
        st.progress(int(round(next_value)))
        st.caption("Model estimate from the road's last three recorded observations; not a live or guaranteed forecast.")
    else:
        st.warning("Not enough observations for this road to calculate an estimate.")

st.divider()
st.markdown('<div class="section-label">02 · Research experiment</div>', unsafe_allow_html=True)
st.subheader("Linear Regression compared with a baseline")
st.write("The baseline assumes the next value will equal the previous recorded congestion value. Both methods are tested on the same later-date holdout set.")
metrics = pd.DataFrame({"Method":["Previous-value baseline","Linear Regression"],"MAE (lower is better)":[base_mae,model_mae],"RMSE (lower is better)":[base_rmse,model_rmse]})
st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
if np.isfinite(improvement):
    if improvement >= 0:
        st.success(f"On this holdout split, Linear Regression reduced MAE by {improvement:.1f}% compared with the baseline.")
    else:
        st.warning(f"On this holdout split, Linear Regression MAE was {abs(improvement):.1f}% higher than the baseline.")
st.caption(f"Chronological split: train on dates before {cutoff.date()}, test on dates from {cutoff.date()} onward. The test is a rolling one-step-ahead evaluation using observed prior values as later test lags.")
plot_df = test[["Date","Road/Intersection Name","Congestion Level"]].copy()
plot_df["Actual congestion"] = actual
plot_df["Linear Regression"] = pred
plot_df["Previous-value baseline"] = baseline
plot_df = plot_df.sort_values("Date").head(500)
fig2 = px.line(plot_df, x="Date", y=["Actual congestion","Linear Regression","Previous-value baseline"], title="Actual and predicted values (first 500 test records)", labels={"value":"Congestion (0–100)","variable":"Series"})
fig2.update_layout(margin=dict(l=10,r=10,t=55,b=10), xaxis_title="Recorded date")
st.plotly_chart(fig2, use_container_width=True)
with st.expander("Inspect and download test predictions"):
    st.dataframe(plot_df[["Date","Road/Intersection Name","Actual congestion","Linear Regression","Previous-value baseline"]].round(2), use_container_width=True, hide_index=True)
    st.download_button("Download test predictions (CSV)", plot_df.to_csv(index=False).encode("utf-8"), file_name="trafficwise_test_predictions.csv", mime="text/csv")

st.divider()
st.markdown('<div class="section-label">03 · Area-level congestion</div>', unsafe_allow_html=True)
st.subheader("Compare average congestion across areas")
area_avg = model_df.groupby("Area Name", as_index=False)["Congestion Level"].mean().sort_values("Congestion Level", ascending=False)
area_avg["Congestion band"] = pd.cut(area_avg["Congestion Level"], bins=[-0.1,35,70,100], labels=["Lower","Moderate","Higher"])
fig3 = px.bar(area_avg, x="Area Name", y="Congestion Level", color="Congestion band", color_discrete_map={"Lower":"#2a9d8f","Moderate":"#e9c46a","Higher":"#e76f51"}, range_y=[0,100], title="Historical mean congestion by area")
fig3.update_layout(margin=dict(l=10,r=10,t=55,b=10), xaxis_title="Area", yaxis_title="Mean congestion (0–100)", legend_title="Congestion band")
st.plotly_chart(fig3, use_container_width=True)
st.markdown("**Area summary**")
st.dataframe(area_avg.rename(columns={"Congestion Level":"Mean congestion (0–100)"}).round({"Mean congestion (0–100)":2}), use_container_width=True, hide_index=True)

with st.expander("Methodology and limitations"):
    st.markdown("""
- **Model:** Linear Regression using the previous three recorded congestion values, day of week, month, and road identity.
- **Baseline:** previous recorded congestion value.
- **Evaluation:** chronological 80/20 split by date; MAE and RMSE are measured on the later-date holdout.
- **Interpretation:** MAE is average absolute prediction error in congestion points; RMSE penalizes larger errors more strongly. Lower is better.
- **Prediction horizon:** next recorded observation. The dataset cadence is not confirmed as hourly, so this is not a minute-by-minute forecast.
- **Data limitations:** verify dataset provenance before making real-world claims. This dashboard does not connect to live traffic feeds.
""")
st.caption("TrafficWise · Urban Traffic Congestion Prediction · Interactive research dashboard")
