"""
config.py — Central configuration for the Traffic Demand Prediction project.

All paths, constants, and shared settings live here so that every notebook
and script pulls from a single source of truth. Changing a path here
propagates everywhere automatically.
"""

import os

# ── Root directory of the project ────────────────────────────────────────────
ROOT_DIR = r"C:\traffic-demand-final"

# ── Raw dataset paths ─────────────────────────────────────────────────────────
DATASETS_DIR   = os.path.join(ROOT_DIR, "DATASETS")
TRAIN_FILE     = os.path.join(DATASETS_DIR, "train.csv")
TEST_FILE      = os.path.join(DATASETS_DIR, "test.csv")
SAMPLE_SUB     = os.path.join(DATASETS_DIR, "sample_submission.csv")

# ── Data-preprocessing notebook and plot directories ─────────────────────────
DATA_PREP_DIR         = os.path.join(ROOT_DIR, "DATA-PREPROCESSING")
PLOTS_DIR             = os.path.join(DATA_PREP_DIR, "plots")
UNIVARIATE_PLOTS_DIR  = os.path.join(PLOTS_DIR, "univariate")
BIVARIATE_PLOTS_DIR   = os.path.join(PLOTS_DIR, "bivariate")
MULTIVARIATE_PLOTS_DIR = os.path.join(PLOTS_DIR, "multivariate")

# ── Output directory (processed datasets, model artefacts, submissions) ───────
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# ── Column name constants ─────────────────────────────────────────────────────
TARGET_COL     = "demand"
ID_COL         = "Index"
GEOHASH_COL    = "geohash"
TIMESTAMP_COL  = "timestamp"
DAY_COL        = "day"

# Categorical columns with low cardinality — suitable for ordinal / label encoding
LOW_CARD_CATS  = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]

# Numerical columns
NUMERIC_COLS   = ["NumberofLanes", "Temperature", "demand"]

# ── Ordinal encoding maps ─────────────────────────────────────────────────────
# RoadType ordered by road capacity / traffic volume: Residential < Street < Highway.
# Using ordinal integers preserves the natural rank ordering for tree-based models.
ROAD_TYPE_MAP = {
    "Residential": 0,
    "Street":      1,
    "Highway":     2,
}

# Weather ordered by severity of impact on driving conditions: Sunny is baseline.
# Snowy represents the most severe disruption and carries the highest ordinal value.
WEATHER_MAP = {
    "Sunny": 0,
    "Foggy": 1,
    "Rainy": 2,
    "Snowy": 3,
}

# LargeVehicles and Landmarks are binary — mapped directly to 0/1 inline in notebooks.
LARGE_VEHICLES_MAP = {"Not Allowed": 0, "Allowed": 1}
LANDMARKS_MAP      = {"No": 0, "Yes": 1}

# ── Preprocessing & Feature Engineering Configs ───────────────────────────────

# Imputation strategies based on univariate analysis
MODE_IMPUTE_COLS   = ["RoadType", "Weather"]
MEDIAN_IMPUTE_COLS = ["Temperature"]

# Encoding strategies
TARGET_ENCODE_COLS = ["geohash"]

# Time-based cyclic features
# timestamp is parsed into decimal_hour, then encoded into sin/cos features.
CYCLIC_FEATURES_MAX = {
    "decimal_hour": 24.0
}

# ── Pipelining directories ────────────────────────────────────────────────────
PIPELINING_DIR = os.path.join(ROOT_DIR, "pipelining")
MODEL_DIR      = os.path.join(PIPELINING_DIR, "models")
PROC_DIR       = os.path.join(PIPELINING_DIR, "processed")
SRC_DIR        = os.path.join(PIPELINING_DIR, "src")

# ── Model settings ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5

# ── Geohash settings ──────────────────────────────────────────────────────────
GEOHASH_PREFIX_LEN = 4

# ── Hour boundary constants derived from bivariate analysis ───────────────────
# Morning peak 7-9, evening peak 17-20 confirmed from dual-peak pattern
PEAK_HOURS     = [(7, 9), (17, 20)]
BUSINESS_HOURS = (8, 17)
NIGHT_HOURS    = (22, 6)

# ── XGBoost best params from previous Optuna run ─────────────────────────────
XGB_PARAMS = {
    "n_estimators":      347,
    "learning_rate":     0.0297,
    "max_depth":         7,
    "subsample":         0.51,
    "colsample_bytree":  0.82,
    "min_child_weight":  7,
    "random_state":      42,
    "verbosity":         0,
    "n_jobs":            -1,
}

# ── LightGBM best params from previous Optuna run ────────────────────────────
LGBM_PARAMS = {
    "n_estimators":      1944,
    "learning_rate":     0.0127,
    "num_leaves":        230,
    "max_depth":         12,
    "subsample":         0.79,
    "colsample_bytree":  0.85,
    "min_child_samples": 5,
    "random_state":      42,
    "verbosity":         -1,
    "n_jobs":            -1,
}

# ── Ridge meta-learner alpha from previous CV run ─────────────────────────────
RIDGE_ALPHA = 2.56

# ── Final 26 features confirmed after VIF check ───────────────────────────────
# 23 original + 3 new spatial features from pygeohash (lat, lon, prefix enc)
FINAL_FEATURES = [
    "geohash_target_enc",
    "latitude",
    "longitude",
    "geohash_prefix_enc",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_peak_hour",
    "is_business_hour",
    "is_night",
    "hour_squared",
    "road_type_ord",
    "NumberofLanes",
    "road_capacity",
    "weather_severity",
    "Temperature",
    "temp_squared",
    "road_combo",
    "hour_x_roadtype",
    "geo_hour_sin",
    "road_x_weather",
    "geo_x_roadtype",
    "geo_x_weather",
    "lanes_x_weather",
    "geo_x_peak",
    "temp_x_weather",
]

# ── Ensure all plot sub-directories exist at import time ──────────────────────
for _dir in [UNIVARIATE_PLOTS_DIR, BIVARIATE_PLOTS_DIR, MULTIVARIATE_PLOTS_DIR, OUTPUT_DIR, MODEL_DIR, PROC_DIR]:
    os.makedirs(_dir, exist_ok=True)

