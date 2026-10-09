import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

BASE = Path(__file__).parent
SITE = json.loads((BASE / "model" / "site_data.json").read_text())
METRICS = json.loads((BASE / "model" / "metrics.json").read_text())
BUNDLE = joblib.load(BASE / "model" / "aqi_model.joblib")
MODEL, MEDIANS, FEATURES = BUNDLE["model"], BUNDLE["medians"], BUNDLE["features"]

app = Flask(__name__)

FIELDS = [  # key, label, unit, hint
    ("PM2.5", "PM2.5", "µg/m³", "fine particles"),
    ("PM10", "PM10", "µg/m³", "coarse particles"),
    ("NO", "NO", "µg/m³", "nitric oxide"),
    ("NO2", "NO₂", "µg/m³", "nitrogen dioxide"),
    ("NOx", "NOx", "ppb", "nitrogen oxides"),
    ("NH3", "NH₃", "µg/m³", "ammonia"),
    ("CO", "CO", "mg/m³", "carbon monoxide"),
    ("SO2", "SO₂", "µg/m³", "sulphur dioxide"),
    ("O3", "O₃", "µg/m³", "ozone"),
    ("Benzene", "Benzene", "µg/m³", ""),
    ("Toluene", "Toluene", "µg/m³", ""),
]

BANDS = [
    ("Good", "0–50", "Minimal impact."),
    ("Satisfactory", "51–100", "Minor breathing discomfort to sensitive people."),
    ("Moderate", "101–200", "Breathing discomfort to people with lung, heart disease, children and older adults."),
    ("Poor", "201–300", "Breathing discomfort to most people on prolonged exposure."),
    ("Very Poor", "301–400", "Respiratory illness on prolonged exposure."),
    ("Severe", "401–500", "Affects healthy people and seriously impacts those with existing diseases."),
]

ADVICE = {
    "Good": ("Air quality is good. Normal outdoor activity is fine.",
             "No special precautions needed."),
    "Satisfactory": ("Acceptable for most people. Enjoy outdoor activity as usual.",
                     "Unusually sensitive people should watch for symptoms during long outdoor exertion."),
    "Moderate": ("Reduce long or heavy exertion outdoors if you notice coughing or throat irritation.",
                 "Keep reliever medication handy. Prefer indoor activity, avoid busy roads."),
    "Poor": ("Avoid prolonged outdoor exertion. Keep windows closed near traffic.",
             "Stay indoors where possible. Consider an N95 mask outdoors."),
    "Very Poor": ("Avoid outdoor exercise. Wear an N95 mask if you must go out.",
                  "Stay indoors, use an air purifier if available, and contact a doctor if symptoms appear."),
    "Severe": ("Stay indoors. Postpone outdoor activity.",
               "Avoid all outdoor exertion, seek medical advice for any breathing difficulty."),
}


def bucket_of(aqi):
    if aqi is None:
        return None
    for limit, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                        (300, "Poor"), (400, "Very Poor"), (10_000, "Severe")]:
        if aqi <= limit:
            return name


@app.template_filter("slug")
def slug(b):
    return (b or "na").lower().replace(" ", "-")


@app.template_filter("num")
def num(v, d=0):
    return "–" if v is None else f"{v:,.{d}f}"


@app.context_processor
def inject():
    return {"meta": SITE["meta"], "bands": BANDS}


def run_prediction(values):
    """values: dict feature -> float|None. returns dict"""
    given = {k: v for k, v in values.items() if v is not None}
    if len(given) < 2 or ("PM2.5" not in given and "PM10" not in given):
        raise ValueError("Enter at least PM2.5 or PM10 plus one more pollutant.")
    row = {f: (values.get(f) if values.get(f) is not None else MEDIANS[f]) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    pred = float(max(MODEL.predict(X)[0], 0.0))
    per_tree = np.array([t.predict(X.values)[0] for t in MODEL.estimators_])
    lo, hi = np.percentile(per_tree, [10, 90])
    b = bucket_of(pred)
    return {
        "aqi": round(pred), "low": round(float(lo)), "high": round(float(hi)), "bucket": b,
        "advice": ADVICE[b][0], "advice_sensitive": ADVICE[b][1],
        "filled_by_default": [f for f in FEATURES if values.get(f) is None],
        "n_given": len(given),
    }


# ------------------------------------------------------------------ pages
@app.route("/")
def index():
    cities = sorted(SITE["cities"].values(), key=lambda c: c["name"])
    ranked = sorted([c for c in cities if c["avg_aqi"] is not None], key=lambda c: -c["avg_aqi"])
    return render_template("index.html", cities=cities, worst=ranked[:5], best=ranked[-5:][::-1])


@app.route("/city/<name>")
def city(name):
    c = SITE["cities"].get(name)
    if not c:
        abort(404)
    return render_template("city.html", c=c)


@app.route("/stations")
def stations():
    states = sorted({s["state"] for s in SITE["stations"]})
    return render_template("stations.html", stations=SITE["stations"], states=states)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, error, form = None, None, {}
    if request.method == "POST":
        vals = {}
        for key, *_ in FIELDS:
            raw = (request.form.get(key) or "").strip()
            form[key] = raw
            try:
                vals[key] = float(raw) if raw != "" else None
            except ValueError:
                error = f"{key}: '{raw}' is not a number."
            if vals.get(key) is not None and vals[key] < 0:
                error = f"{key} cannot be negative."
        form["sensitive"] = request.form.get("sensitive") == "on"
        if not error:
            try:
                result = run_prediction(vals)
            except ValueError as e:
                error = str(e)
    return render_template("predict.html", fields=FIELDS, result=result, error=error, form=form)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    vals = {f: (float(data[f]) if data.get(f) not in (None, "") else None) for f in FEATURES}
    try:
        return jsonify(run_prediction(vals))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model")
def model_page():
    return render_template("model.html", m=METRICS)


@app.errorhandler(404)
def nf(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
