"""
prepare.py  -  run ONCE (and again whenever you add/replace a CSV in data/).

Reads every CSV it finds in data/:
    stations.csv, city_day.csv, station_day.csv      (required)
    city_hour.csv, station_hour.csv                  (optional - used if present)
Writes:
    model/aqi_model.joblib   trained Random Forest + medians
    model/metrics.json       model comparison table, feature importance
    model/site_data.json     everything the website pages show
"""
import json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA = Path("data")
OUT = Path("model")
OUT.mkdir(exist_ok=True)

FEATURES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
BUCKETS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
           (300, "Poor"), (400, "Very Poor"), (10_000, "Severe")]


def bucket(aqi):
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return None
    for limit, name in BUCKETS:
        if aqi <= limit:
            return name


def find(name):
    """case-insensitive file lookup inside data/"""
    for p in DATA.glob("*.csv"):
        if p.name.lower() == name.lower():
            return p
    return None


def r(x, n=1):
    return None if x is None or pd.isna(x) else round(float(x), n)


# ------------------------------------------------------------------ load
paths = {k: find(k + ".csv") for k in ["stations", "city_day", "station_day", "city_hour", "station_hour"]}
for k in ["stations", "city_day", "station_day"]:
    if paths[k] is None:
        sys.exit(f"ERROR: data/{k}.csv not found. Put your CSV files inside the data/ folder.")

stations = pd.read_csv(paths["stations"])
city_day = pd.read_csv(paths["city_day"], parse_dates=["Date"])
station_day = pd.read_csv(paths["station_day"], parse_dates=["Date"])
print("loaded:", {k: (v.name if v else "MISSING (optional)") for k, v in paths.items()})

city_state = stations.groupby("City")["State"].agg(lambda s: s.mode().iat[0]).to_dict()

# ------------------------------------------------------------------ model
# city_day + station_day are stacked together -> far more rows than city_day alone
cols = ["Date"] + FEATURES + ["AQI"]
all_rows = pd.concat([city_day[cols], station_day[cols]], ignore_index=True).dropna(subset=["AQI"])
print("training rows:", len(all_rows))

X, y = all_rows[FEATURES], all_rows["AQI"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
medians = X_tr.median()
X_tr_i, X_te_i = X_tr.fillna(medians), X_te.fillna(medians)


def score(name, yt, yp):
    mse = mean_squared_error(yt, yp)
    return {"model": name, "MAE": r(mean_absolute_error(yt, yp), 4), "MSE": r(mse, 4),
            "RMSE": r(np.sqrt(mse), 4), "R2": r(r2_score(yt, yp), 4)}


def make_rf():
    # min_samples_leaf + max_samples keep the saved model file reasonably small
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_samples=0.6,
                                 n_jobs=-1, random_state=42)


t0 = time.time()
results, models = [], {}
for name, m in [("Linear Regression", LinearRegression()),
                ("Decision Tree", DecisionTreeRegressor(random_state=42, min_samples_leaf=3)),
                ("Random Forest", make_rf())]:
    m.fit(X_tr_i, y_tr)
    results.append(score(name, y_te, m.predict(X_te_i)))
    models[name] = m
    print(results[-1])
print(f"trained in {time.time() - t0:.0f}s")

# time-aware check (train on the past, test on the future) - honest generalisation number
cut = pd.Timestamp("2019-07-01")
past, future = all_rows[all_rows.Date < cut], all_rows[all_rows.Date >= cut]
time_result = None
if len(past) > 1000 and len(future) > 1000:
    med_p = past[FEATURES].median()
    rf_t = make_rf().fit(past[FEATURES].fillna(med_p), past["AQI"])
    time_result = score("Random Forest (time split: train <2019-07, test >=2019-07)",
                        future["AQI"], rf_t.predict(future[FEATURES].fillna(med_p)))
    print(time_result)

best = models["Random Forest"]
importance = sorted(zip(FEATURES, best.feature_importances_), key=lambda t: -t[1])
joblib.dump({"model": best, "medians": medians.to_dict(), "features": FEATURES},
            OUT / "aqi_model.joblib", compress=3)
(OUT / "metrics.json").write_text(json.dumps({
    "results": results, "time_split": time_result,
    "importance": [[f, round(float(v), 4)] for f, v in importance],
    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
    "sources": ["city_day", "station_day"],
}, indent=1))

# ------------------------------------------------------------------ site data
def last_valid(df, key):
    d = df.dropna(subset=["AQI"]).sort_values("Date")
    return d.groupby(key).tail(1).set_index(key)


# ---- cities
last_c = last_valid(city_day, "City")
cities = {}
for city, g in city_day.groupby("City"):
    gv = g.dropna(subset=["AQI"])
    monthly = gv.groupby(gv.Date.dt.to_period("M"))["AQI"].mean()
    yearly = gv.groupby(gv.Date.dt.year)["AQI"].mean()
    worst = gv.loc[gv.AQI.idxmax()] if len(gv) else None
    cities[city] = {
        "name": city, "state": city_state.get(city, ""),
        "latest_date": last_c.loc[city, "Date"].strftime("%Y-%m-%d") if city in last_c.index else None,
        "latest_aqi": r(last_c.loc[city, "AQI"], 0) if city in last_c.index else None,
        "avg_aqi": r(gv.AQI.mean(), 0), "records": int(len(gv)),
        "bucket": bucket(last_c.loc[city, "AQI"]) if city in last_c.index else None,
        "monthly": [[str(p), r(v, 0)] for p, v in monthly.items()],
        "yearly": [[int(k), r(v, 0)] for k, v in yearly.items()],
        "pollutants": {f: r(g[f].mean(), 2) for f in FEATURES},
        "bucket_counts": gv.AQI.map(bucket).value_counts().to_dict(),
        "worst": {"date": worst.Date.strftime("%Y-%m-%d"), "aqi": r(worst.AQI, 0)} if worst is not None else None,
        "hour_profile": None, "stations": [],
    }

# ---- stations
last_s = last_valid(station_day, "StationId")
sd_stats = station_day.dropna(subset=["AQI"]).groupby("StationId")["AQI"].agg(["mean", "count"])
station_list = []
for _, s in stations.iterrows():
    sid = s.StationId
    has = sid in last_s.index
    row = {"id": sid, "name": s.StationName, "city": s.City, "state": s.State,
           "status": s.Status if isinstance(s.Status, str) else "Unknown",
           "latest_date": last_s.loc[sid, "Date"].strftime("%Y-%m-%d") if has else None,
           "latest_aqi": r(last_s.loc[sid, "AQI"], 0) if has else None,
           "avg_aqi": r(sd_stats.loc[sid, "mean"], 0) if sid in sd_stats.index else None,
           "records": int(sd_stats.loc[sid, "count"]) if sid in sd_stats.index else 0,
           "hourly_records": None}
    row["bucket"] = bucket(row["latest_aqi"])
    station_list.append(row)
    # attach to a city page (city names in stations.csv can differ from city_day, so match loosely)
    if s.City in cities:
        cities[s.City]["stations"].append(sid)

# ---- hourly files (optional, read in chunks so big files don't eat RAM)
hourly_info = {"city_hour": False, "station_hour": False}
if paths["city_hour"]:
    acc = {}
    for ch in pd.read_csv(paths["city_hour"], usecols=lambda c: c in {"City", "Datetime", "PM2.5", "AQI"}, chunksize=300_000):
        ch["h"] = ch["Datetime"].str.slice(11, 13).astype(int)
        for (city, h), g in ch.groupby(["City", "h"]):
            a = acc.setdefault((city, h), [0.0, 0, 0.0, 0])
            a[0] += g["PM2.5"].sum(); a[1] += g["PM2.5"].count()
            a[2] += g["AQI"].sum(); a[3] += g["AQI"].count()
    for city in cities:
        prof = []
        for h in range(24):
            a = acc.get((city, h))
            prof.append([r(a[0] / a[1]) if a and a[1] else None, r(a[2] / a[3], 0) if a and a[3] else None] if a else [None, None])
        cities[city]["hour_profile"] = prof
    hourly_info["city_hour"] = True
    print("city_hour.csv processed")

if paths["station_hour"]:
    cnt, tot, cntA, lastts = {}, {}, {}, {}
    for ch in pd.read_csv(paths["station_hour"], usecols=lambda c: c in {"StationId", "Datetime", "AQI"}, chunksize=500_000):
        g = ch.groupby("StationId")
        for sid, n in g.size().items():
            cnt[sid] = cnt.get(sid, 0) + int(n)
        for sid, v in g["AQI"].sum().items():
            tot[sid] = tot.get(sid, 0.0) + float(v)
        for sid, v in g["AQI"].count().items():
            cntA[sid] = cntA.get(sid, 0) + int(v)
        for sid, v in g["Datetime"].max().items():
            lastts[sid] = max(lastts.get(sid, ""), v)
    for row in station_list:
        sid = row["id"]
        if sid in cnt:
            row["hourly_records"] = cnt[sid]
            row["hourly_avg_aqi"] = r(tot[sid] / cntA[sid], 0) if cntA.get(sid) else None
            row["last_hour"] = lastts.get(sid)
    hourly_info["station_hour"] = True
    print("station_hour.csv processed")

# station name -> city page
by_id = {s["id"]: s for s in station_list}
for c in cities.values():
    c["stations"] = [by_id[i] for i in c["stations"]]

dates = pd.concat([city_day.Date, station_day.Date])
site = {
    "meta": {"date_from": dates.min().strftime("%d %b %Y"), "date_to": dates.max().strftime("%d %b %Y"),
             "n_cities": len(cities), "n_stations": len(station_list),
             "n_city_days": int(len(city_day)), "n_station_days": int(len(station_day)),
             "hourly": hourly_info, "built": time.strftime("%d %b %Y %H:%M")},
    "cities": cities, "stations": station_list,
}
(OUT / "site_data.json").write_text(json.dumps(site, separators=(",", ":")))
print("done ->", OUT)
