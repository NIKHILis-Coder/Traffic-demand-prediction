import nbformat
from nbclient import NotebookClient
import os

nb_path = r'c:\traffic-demand-final\pipelining\notebooks\step07_interaction_features.ipynb'
print(f"Fixing {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("['road_combo_enc']", "['road_combo']")
        cell.source = cell.source.replace("road_combo_enc", "road_combo")
nbformat.write(nb, nb_path)
print("Fixed step07.")

notebooks_to_run = [
    r'c:\traffic-demand-final\pipelining\notebooks\step07_interaction_features.ipynb',
    r'c:\traffic-demand-final\pipelining\notebooks\step08_weather_features.ipynb',
    r'c:\traffic-demand-final\pipelining\notebooks\step09_vif_check.ipynb',
    r'c:\traffic-demand-final\pipelining\notebooks\step10_model_training.ipynb'
]

for nb_path in notebooks_to_run:
    print(f"Executing {nb_path}...")
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=-1)
    try:
        client.execute()
        nbformat.write(nb, nb_path)
        print(f"Successfully executed {os.path.basename(nb_path)}")
    except Exception as e:
        print(f"Failed to execute {os.path.basename(nb_path)}")
        import traceback
        traceback.print_exc()
        break

print("Script execution complete.")
