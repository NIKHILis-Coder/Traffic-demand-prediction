# Traffic Demand Prediction

Predict traffic demand (a float between 0 and 1) for a road segment at a given
15-minute time slot, given its location, time, road type, and weather. Built
as an end-to-end ML project: EDA → feature engineering → model comparison →
stacked ensemble.

**Final result:** stacked ensemble (XGBoost + LightGBM + CatBoost → Ridge)
achieves **RMSE 0.0263 / R² 0.966** on out-of-fold predictions, a 33% RMSE
improvement over a manually-tuned single-model baseline (0.0395). See
`reports/Traffic_Demand_Approach_Document.txt` for the full writeup and
`INTERVIEW_PREP.md` for a detailed walkthrough of every decision.

## Data

11 raw columns, 77,299 training rows / 41,778 test rows:

| Column | Type | Notes |
|---|---|---|
| geohash | categorical | 1,249 unique location cells |
| day | int | only 2 distinct values — not a real time series |
| timestamp | string `H:MM` | 96 fifteen-minute slots/day |
| RoadType | categorical | Highway / Street / Residential (600 missing) |
| NumberofLanes | int | 1–5 |
| LargeVehicles | binary | Allowed / Not Allowed |
| Landmarks | binary | Yes / No |
| Temperature | float | °C (2,495 missing) |
| Weather | categorical | Sunny / Foggy / Rainy / Snowy (797 missing) |
| demand | float, target | 0–1, right-skewed, mean 0.094 |

## Project structure

```
data/
  raw/                train.csv, test.csv, sample_submission.csv
  processed/          intermediate CSVs written after each pipeline step
notebooks/
  eda/                4 exploratory notebooks (raw -> univariate -> bivariate -> multivariate) + plots/
  pipeline/           11 numbered notebooks, step01 (cleaning) -> step11 (prediction)
src/
  config.py           paths, encoding maps, model hyperparameters — single source of truth
  utils.py            seed_everything(), rmse(), r2()
models/               trained model artifacts + feature importance plots
reports/              submission.csv, prediction plot, approach document
requirements.txt
```

## Pipeline

| Step | What it does |
|---|---|
| 01 | Drop `Index`, parse `timestamp` → `hour`/`minute`, mode-impute `RoadType`/`Weather` (fit on train only) |
| 02 | Impute `Temperature` with the geohash-group median (train-derived), falling back to the global median |
| 03 | Ordinal/binary encode categoricals |
| 04 | Decode `geohash` → lat/lon via `pygeohash`; leave-one-out target encoding of `geohash` |
| 05 | Cyclical hour features (`sin`/`cos`), peak/business/night flags, `hour²` |
| 06 | `road_capacity` (type × lanes), `road_combo` (vehicles × landmarks) |
| 07 | 8 interaction features, chosen from multivariate EDA findings |
| 08 | `temp²`, trim to the final 26-feature set |
| 09 | VIF multicollinearity check (diagnostic — see note below) |
| 10 | Optuna-tuned XGBoost + LightGBM + CatBoost, 5-fold out-of-fold stacking with a Ridge meta-learner |
| 11 | Load saved models, predict on test, clip to `[0, ∞)`, write submission |

Every encoder/imputer is fit on train only and applied to test — no target
leakage between splits. `geohash` uses genuine leave-one-out encoding (each
row's encoding excludes its own value) rather than a plain group mean, which
would leak.

**On the VIF step:** 16 of 23 extended features exceed VIF 10 — expected,
since most are engineered products of other retained features
(`road_capacity = road_type × lanes`, `geo_x_roadtype = geohash_enc ×
road_type`, etc.). All 26 features are kept anyway: XGBoost/LightGBM/CatBoost
split on one feature at a time and are invariant to linear collinearity; VIF
only threatens the Ridge meta-learner, which never sees these 26 raw features
(it only sees the 3 base models' predictions). The full reasoning is in
`notebooks/pipeline/step09_vif_check.ipynb`.

## Results

| Model | OOF RMSE | OOF R² |
|---|---|---|
| XGBoost | 0.0309 | 95.26% |
| LightGBM | 0.0274 | 96.27% |
| CatBoost | 0.0274 | 96.29% |
| **Stacked (Ridge)** | **0.0263** | **96.57%** |

All metrics are out-of-fold (5-fold CV) — no train-set leakage into the
reported numbers.

## Running it

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order: `notebooks/pipeline/step01...` through `step11`.
Each step reads the previous step's output from `data/processed/` and writes
its own. `step10_model_training.ipynb` runs the Optuna hyperparameter search
(200 trials × 3 models — slow) and saves `models/best_params.json`;
`step10_part2_training.ipynb` reloads those params and does the actual
5-fold stacking + final model fit (kept as a separate notebook so the final
training run doesn't require re-doing the search every time).

`src/config.py` resolves all paths relative to its own location, so the
project runs from any clone location without editing paths.
