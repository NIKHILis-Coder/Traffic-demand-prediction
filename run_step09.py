import nbformat
from nbclient import NotebookClient
import os

nb_path = r'c:\traffic-demand-final\pipelining\notebooks\step09_vif_check.ipynb'
print(f"Fixing {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'corr_matrix =' in cell.source:
        cell.source = """import seaborn as sns
import matplotlib.pyplot as plt

# Load step07 data to access LargeVehicles and Landmarks which were dropped in step08
df_07 = pd.read_csv(os.path.join(PROC_DIR, 'train_step07.csv'))
corr_features = ['road_type_ord', 'NumberofLanes', 'road_capacity', 'LargeVehicles', 'Landmarks']
corr_matrix = df_07[corr_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Road Features')
os.makedirs(MULTIVARIATE_PLOTS_DIR, exist_ok=True)
plt.savefig(os.path.join(MULTIVARIATE_PLOTS_DIR, 'road_features_correlation.png'))
plt.show()"""
nbformat.write(nb, nb_path)
print("Fixed step09.")

notebooks_to_run = [
    r'c:\traffic-demand-final\pipelining\notebooks\step09_vif_check.ipynb'
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

print("Pipeline execution script complete.")
