import nbformat

nb = nbformat.read('pipelining/notebooks/step10_part2_training.ipynb', 4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and 'outputs' in c:
        print(f"--- Cell {i} Outputs ---")
        for out in c.outputs:
            if 'text' in out:
                print(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                print(out['data']['text/plain'])
