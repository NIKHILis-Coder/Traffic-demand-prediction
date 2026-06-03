import nbformat
from nbclient import NotebookClient
import os

nb_path = r'c:\traffic-demand-final\pipelining\notebooks\step10_model_training.ipynb'
print(f"Fixing {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Fix for XGBoost Objective
        if 'early_stopping_rounds=30' in cell.source and 'model = XGBRegressor' in cell.source:
            cell.source = cell.source.replace('model = XGBRegressor(**params)', 'model = XGBRegressor(**params, early_stopping_rounds=30)')
            cell.source = cell.source.replace('early_stopping_rounds=30,', '')
        
        # Fix for XGBoost KFold loop
        if 'early_stopping_rounds=50' in cell.source and 'xgb = XGBRegressor' in cell.source:
            cell.source = cell.source.replace('xgb = XGBRegressor(**xgb_best_params)', 'xgb = XGBRegressor(**xgb_best_params, early_stopping_rounds=50)')
            cell.source = cell.source.replace('early_stopping_rounds=50, ', '')

nbformat.write(nb, nb_path)
print("Fixed early_stopping_rounds for XGBoost.")

print("Executing step 10...")
client = NotebookClient(nb, timeout=-1)
client.execute()
nbformat.write(nb, nb_path)
print("Successfully executed step10_model_training.ipynb")
