import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(r"c:\traffic-demand-final\pipelining\src")
from config import FINAL_FEATURES, MODEL_DIR, OUTPUT_DIR, SAMPLE_SUB

def main():
    print("Loading test data...")
    test_df = pd.read_csv(r"c:\traffic-demand-final\pipelining\processed\test_step08.csv")
    X_test = test_df[FINAL_FEATURES]
    
    print("Loading models...")
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'best_xgb_model.pkl'))
    lgb_model = joblib.load(os.path.join(MODEL_DIR, 'best_lgbm_model.pkl'))
    cat_model = joblib.load(os.path.join(MODEL_DIR, 'best_catboost_model.pkl'))
    meta_learner = joblib.load(os.path.join(MODEL_DIR, 'meta_learner.pkl'))
    
    print("Generating base predictions...")
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)
    cat_preds = cat_model.predict(X_test)
    
    print("Stacking predictions...")
    test_stack = np.column_stack((xgb_preds, lgb_preds, cat_preds))
    
    print("Generating final predictions...")
    final_preds = meta_learner.predict(test_stack)
    
    print("Saving submission...")
    sub_df = pd.read_csv(SAMPLE_SUB)
    sub_df['demand'] = final_preds
    sub_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print("Successfully generated submission.csv in output directory.")

if __name__ == "__main__":
    main()
