# Air Quality India – AQI Website (BABMH203)

Flask website with Cities, City Detail (charts), Stations (search/filter), Predict AQI, and About the Model pages.

## Run it locally (Windows)

1. Install Python 3.10+.

2. Put the required CSV files in the `data/` folder:

   * `stations.csv`
   * `city_day.csv`
   * `station_day.csv`
   * `city_hour.csv` and `station_hour.csv` when available

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Build the model and site data:

   ```bash
   python prepare.py
   ```

5. Start the website:

   ```bash
   python app.py
   ```

6. Open http://127.0.0.1:5000

If you add or replace a CSV later, run `python prepare.py` again and restart the app.

## Deploy online with Render

* Build command: `pip install -r requirements.txt`
* Start command: `gunicorn app:app`

The `data/` folder is excluded from Git by `.gitignore`. The generated `model/` files are included in the repository.

## Project files

* `prepare.py` — reads CSV files, trains models, and writes generated files to `model/`
* `app.py` — Flask website and prediction routes
* `templates/` — HTML pages
* `static/` — CSS, JavaScript, and chart assets
