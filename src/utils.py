import numpy as np
import random
import os
from sklearn.metrics import mean_squared_error, r2_score

def seed_everything(seed=42):
    # random and numpy are two separate RNG streams (numpy sampling, pandas
    # .sample() etc. use np.random; anything using stdlib random doesn't) —
    # both need seeding. PYTHONHASHSEED matters for hash-order-dependent code
    # (e.g. set/dict iteration) but only takes effect on process start, so it's
    # set here mainly for anyone re-running this as a script rather than in a
    # live notebook kernel. Model-level randomness (XGBoost/LightGBM/CatBoost)
    # is seeded separately via each model's own random_state param.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
