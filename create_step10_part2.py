import nbformat
import os

nb_path = r'c:\traffic-demand-final\pipelining\notebooks\step10_model_training.ipynb'
out_path = r'c:\traffic-demand-final\pipelining\notebooks\step10_part2_training.ipynb'

nb = nbformat.read(nb_path, as_version=4)

new_nb = nbformat.v4.new_notebook()
new_nb.cells.append(nb.cells[0]) # imports and data load

code = """
import json, os
print("Loading best parameters...")
with open(os.path.join(MODEL_DIR, 'best_params.json')) as f:
    p = json.load(f)
xgb_best_params = p['xgb']
lgb_best_params = p['lgb']
cat_best_params = p['cat']
print("Loaded tuned parameters successfully.")
"""
load_cell = nbformat.v4.new_code_cell(code)
new_nb.cells.append(load_cell)

# Append remaining cells starting from cell 5
new_nb.cells.extend(nb.cells[5:])

nbformat.write(new_nb, out_path)
print(f"Created {out_path}")
