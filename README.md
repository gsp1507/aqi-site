# Air Quality India – AQI website (BABMH203)

Flask website. Pages: Cities, City detail (charts), Stations (search/filter), Predict AQI, About the model.

## Run it on your laptop (Windows)

1. Install Python 3.10+ from python.org (tick "Add Python to PATH").
2. Unzip this folder. Open a terminal inside it (Shift + right-click > "Open in Terminal").
3. Put ALL your CSV files in the `data/` folder:
   stations.csv, city_day.csv, station_day.csv  (+ city_hour.csv, station_hour.csv when you have them)
4. Install packages:        pip install -r requirements.txt
5. Build model + site data: python prepare.py      (takes 1-3 minutes; station_hour.csv adds a few more)
6. Start the site:          python app.py
7. Open http://127.0.0.1:5000

Added or replaced a CSV later? Run `python prepare.py` again, restart `python app.py`.

## Put it online (free, Render.com)

1. Run steps 4-5 above so the `model/` folder exists.
2. `pip freeze` -> find your scikit-learn version -> write it in requirements.txt (e.g. scikit-learn==1.5.2).
   Then run prepare.py again if you changed versions, so the saved model matches.
3. Make a GitHub repo containing everything EXCEPT `data/` (already in .gitignore). The `model/` folder (~40 MB) must be included.
4. render.com > New > Web Service > connect the repo.
   Build command: pip install -r requirements.txt
   Start command: gunicorn app:app
5. First load after idle takes ~30-60 s on the free plan.

## Files
- prepare.py   reads every CSV in data/, trains Linear / Decision Tree / Random Forest, writes model/
- app.py       the website (routes, prediction)
- templates/   HTML pages        static/   CSS, JS, Chart.js (bundled, works offline)
