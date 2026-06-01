# Traffic Demand Prediction

This repository contains an end-to-end machine learning pipeline to predict traffic demand. It encompasses extensive Exploratory Data Analysis (EDA), robust feature engineering, target encoding, modeling, and evaluation.

## 📂 Project Structure

```text
├── DATA-PREPROCESSING/
│   ├── bivariate_analysis.ipynb     # Analysis between two variables
│   ├── multivariate_analysis.ipynb  # Analysis of multiple variables interacting
│   ├── raw_data_analysis.ipynb      # Initial inspection of the raw datasets
│   ├── univariate_analysis.ipynb    # Distribution and summary of single variables
│   └── plots/                       # Generated EDA plots (univariate, bivariate, etc.)
│
├── DATASETS/
│   ├── train.csv                    # Training data 
│   ├── test.csv                     # Testing data for predictions
│   └── sample_submission.csv        # Format for final submission
│
├── pipelining/
│   ├── notebooks/
│   │   ├── step01_basic_cleaning.ipynb            # Data cleaning and type casting
│   │   ├── step02_temperature_imputation.ipynb    # Handling missing values
│   │   ├── step03_encoding.ipynb                  # Categorical encoding
│   │   ├── step04_geohash_target_encoding.ipynb   # Geospatial target encoding & smoothing
│   │   ├── step05_temporal_features.ipynb         # Time-based feature generation
│   │   ├── step06_road_features.ipynb             # Road specific spatial features
│   │   ├── step07_interaction_features.ipynb      # Interaction term creation
│   │   ├── step08_weather_features.ipynb          # Weather specific feature engineering
│   │   ├── step09_vif_check.ipynb                 # Multicollinearity check (Variance Inflation Factor)
│   │   ├── step10_model_training.ipynb            # Training machine learning models
│   │   └── step11_prediction.ipynb                # Generating final submission file
│   │
│   ├── processed/                   # Intermediate CSV files created at each pipeline step
│   └── src/
│       ├── config.py                # Pipeline configurations and constants
│       └── utils.py                 # Reusable utility functions
│
└── output/
    └── submission.csv               # Final prediction file for submission
```

## 🚀 Getting Started

### Prerequisites

You need Python 3 installed. It is recommended to use a virtual environment. The repository already has a `mlpr` virtual environment directory initialized.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NIKHILis-Coder/Traffic-demand-prediction.git
   cd Traffic-demand-prediction
   ```

2. **Activate the virtual environment (Windows):**
   ```bash
   .\mlpr\Scripts\activate
   ```

3. **Install Dependencies:**
   Make sure you have standard data science libraries installed such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, and `pygeohash`.

### Executing the Pipeline

The project is designed to be executed sequentially via Jupyter Notebooks.
Start Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook
```
Navigate to `pipelining/notebooks/` and execute the steps in numerical order, starting from `step01_basic_cleaning.ipynb` through to `step11_prediction.ipynb`. Intermediate datasets are automatically saved to `pipelining/processed/` after each step.

## 🛠️ Methodology Highlights

- **Spatial Analytics**: Leverages `pygeohash` to decode categorical spatial regions into highly granular coordinates.
- **Leave-One-Out Target Encoding**: Implements robust LOO encoding techniques on the training data to mitigate data leakage, utilizing an elegant multi-level fallback mechanism for unseen geospatial categories in test data.
- **Feature Engineering**: Heavy focus on deriving temporal trends, weather impacts, and interactions to feed the final ML models.
- **VIF Checking**: Includes explicit steps to resolve multicollinearity through Variance Inflation Factor analysis before hitting the final modeling phases.
