import nbformat
from nbclient import NotebookClient

nb_path = r'C:\traffic-demand-final\pipelining\notebooks\step11_prediction.ipynb'
print(f"Loading {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

print("Executing step 11...")
client = NotebookClient(nb, timeout=-1, kernel_name='python3')
client.execute()

nbformat.write(nb, nb_path)
print("Successfully executed step11_prediction.ipynb")
