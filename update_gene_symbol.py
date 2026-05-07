import os
import glob
import re

files_to_check = glob.glob('dataset*.py') + ['build_graph.py', 'interpret.py', 'check_headers.py']

for f in files_to_check:
    if not os.path.exists(f):
        continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        
    new_content = content.replace('final_gene.csv', 'final_gene_symbol.csv')
    
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Updated {f}")
