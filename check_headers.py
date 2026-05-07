import pandas as pd

g = pd.read_csv("data_final/final_gene_symbol.csv", nrows=1)
m = pd.read_csv("data_final/final_mirna.csv", nrows=1)
c = pd.read_csv("data_final/final_methylation.csv", nrows=1)

with open("scratch_headers.txt", "w") as f:
    f.write("Gene: " + ", ".join(g.columns[:5].tolist()) + "\n")
    f.write("miRNA: " + ", ".join(m.columns[:5].tolist()) + "\n")
    f.write("Methyl: " + ", ".join(c.columns[:5].tolist()) + "\n")
