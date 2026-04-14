import os
import sys
import numpy as np
import torch
import csv
import config

try:
    from torch_geometric.data import HeteroData
except ImportError:
    print("WARNING: torch_geometric chưa được cài đặt. Chỉ lưu edge_index_dict.")
    HeteroData = None

def get_base_id(ensembl_id):
    """Strip version number from Ensembl ID, e.g. ENSG00000000003.15 -> ENSG00000000003"""
    return ensembl_id.split('.')[0]

def build_hetero_graph(data_dir=config.GI_FINAL_DIR, save_path=None):
    if save_path is None:
        save_path = os.path.join(config.BASE_DIR, "data_graph", "hetero_graph.pt")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("\n--- 1. READING FEATURES HEADERS ---")
    # Read just the columns
    with open(os.path.join(data_dir, "final_gene.csv"), "r") as f:
        gene_cols = f.readline().strip().split(',')[1:] # skip Patient ID
    with open(os.path.join(data_dir, "final_mirna.csv"), "r") as f:
        mirna_cols = f.readline().strip().split(',')[1:]
    with open(os.path.join(data_dir, "final_methylation.csv"), "r") as f:
        cpg_cols = f.readline().strip().split(',')[1:]
        
    print(f"Loaded {len(gene_cols)} Genes, {len(mirna_cols)} miRNAs, {len(cpg_cols)} CpGs.")
    
    # Create mapping indices
    gene_map = {name: i for i, name in enumerate(gene_cols)}
    gene_base_map = {get_base_id(name): i for i, name in enumerate(gene_cols)}
    mirna_map = {name: i for i, name in enumerate(mirna_cols)}
    cpg_map = {name: i for i, name in enumerate(cpg_cols)}
    
    print("\n--- 2. LOADING ALIASES & BUILDING PPI (GENE-GENE) ---")
    aliases_path = os.path.join(config.BASE_DIR, "Heterogeneous_Graph", "9606.protein.aliases.v12.0.txt")
    
    ensp_to_ensg = {}
    symbol_to_ensg = {}
    if os.path.exists(aliases_path):
        print("Parsing STRING aliases...")
        with open(aliases_path, "r", encoding="utf-8") as f:
            next(f) # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ensp = parts[0]
                    alias = parts[1]
                    source = parts[2]
                    
                    if source == "Ensembl_gene":
                        ensp_to_ensg[ensp] = alias
                    if source.startswith("Ensembl_HGNC"):
                        if alias not in symbol_to_ensg:
                            symbol_to_ensg[alias] = []
                        # symbol_to_ensg maps Hugo to all possible ensp first
                        # We will map symbol to ENSG later
                        pass

    # Better to use GTF for Symbol -> ENSG if available, but let's parse aliases
    # For symbol mapping, we can just map symbol -> ensp -> ensembl_gene
    if os.path.exists(aliases_path):
        with open(aliases_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ensp = parts[0]
                    alias = parts[1]
                    source = parts[2]
                    if source.startswith("Ensembl_HGNC"):
                        if ensp in ensp_to_ensg:
                            ensg = ensp_to_ensg[ensp]
                            symbol_to_ensg[alias] = ensg
    
    # Load STRING Link
    links_path = os.path.join(config.BASE_DIR, "Heterogeneous_Graph", "9606.protein.links.v12.0.txt")
    gene_gene_edges = []
    if os.path.exists(links_path):
        print("Parsing STRING links...")
        with open(links_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    p1, p2, score = parts[0], parts[1], int(parts[2])
                    if score >= 700:
                        g1 = ensp_to_ensg.get(p1)
                        g2 = ensp_to_ensg.get(p2)
                        if g1 and g2:
                            if g1 in gene_base_map and g2 in gene_base_map:
                                gene_gene_edges.append((gene_base_map[g1], gene_base_map[g2]))
        print(f"Extracted {len(gene_gene_edges)} gene-gene edges.")
    else:
        print("WARNING: STRING links not found.")
        
    print("\n--- 3. miRNA-GENE EDGES ---")
    mti_path = os.path.join(config.BASE_DIR, "Heterogeneous_Graph", "hsa_MTI.csv")
    mirna_gene_edges = []
    if os.path.exists(mti_path):
        print("Parsing miRTarBase...")
        # miRNAs in our data are hsa-let-7a-1, hsa-mir-1-1
        # MTI uses hsa-let-7a-5p or hsa-miR-1
        # We will do a generic substring/prefix matching if needed
        # Prepare mirna prefixes
        mir_prefixes = {}
        for m in mirna_cols:
            # handle both -1, -5p suffixes
            prefix = m.lower().replace("-mir-", "-mir-")
            if prefix.endswith("-1") or prefix.endswith("-2") or prefix.endswith("-3"):
                prefix = prefix[:-2]
            mir_prefixes[prefix] = mirna_map[m]
            
        with open(mti_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header_idx = None
            for i, row in enumerate(reader):
                if row and len(row) > 3 and row[1] == "miRNA":
                    header_idx = i
                    header = row
                    break
            
            if header_idx is not None:
                mi_col = header.index("miRNA")
                tgt_col = header.index("Target Gene")
                
                with open(mti_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for _ in range(header_idx + 1):
                        next(reader)
                    for row in reader:
                        if len(row) > max(mi_col, tgt_col):
                            mir = str(row[mi_col]).lower()
                            target = str(row[tgt_col])
                            
                            # find mapped mirna
                            if mir.endswith("-5p") or mir.endswith("-3p"):
                                mir = mir[:-3]
                                
                            mir_idx = mir_prefixes.get(mir)
                            ensg = symbol_to_ensg.get(target)
                            gene_idx = gene_base_map.get(ensg) if ensg else None
                            
                            if mir_idx is not None and gene_idx is not None:
                                mirna_gene_edges.append((mir_idx, gene_idx))
        
        # Remove duplicates
        mirna_gene_edges = list(set(mirna_gene_edges))
        print(f"Extracted {len(mirna_gene_edges)} miRNA-gene edges.")
    else:
        print("WARNING: miRTarBase not found.")
        
    print("\n--- 4. CPG-GENE EDGES ---")
    manifest_path = os.path.join(config.ANNOTATION_DIR, "HumanMethylation450_manifest.csv.csv")
    if not os.path.exists(manifest_path):
        # Fallback to the original annotation name in case
        manifest_path = os.path.join(config.ANNOTATION_DIR, "HumanMethylation450_manifest.csv")
        
    cpg_gene_edges = []
    if os.path.exists(manifest_path):
        print("Parsing Methylation Manifest...")
        # Since the first lines are metadata, we can skip until [Assay] or [Heading]
        try:
            # Just read chunks using pandas since read_csv skip rows might be tricky
            # the column is IlmnID and UCSC_RefGene_Name
            # Let's read by csv reader
            import csv
            with open(manifest_path, 'r', errors='ignore') as f:
                reader = csv.reader(f)
                header_idx = None
                for i, row in enumerate(reader):
                    if row and row[0] == "IlmnID":
                        header_idx = i
                        header = row
                        break
            
            if header_idx is not None:
                cpg_idx_col = header.index("IlmnID")
                gene_idx_col = header.index("UCSC_RefGene_Name")
                
                with open(manifest_path, 'r', errors='ignore') as f:
                    reader = csv.reader(f)
                    for i in range(header_idx + 1):
                        next(reader)
                    for row in reader:
                        if len(row) > max(cpg_idx_col, gene_idx_col):
                            cpg = row[cpg_idx_col]
                            genes = row[gene_idx_col]
                            if cpg in cpg_map and genes:
                                # A probe can target multiple genes, separated by semicolon
                                for gene_symbol in set(genes.split(';')):
                                    ensg = symbol_to_ensg.get(gene_symbol)
                                    g_id = gene_base_map.get(ensg) if ensg else None
                                    if g_id is not None:
                                        cpg_gene_edges.append((cpg_map[cpg], g_id))
                                        
                cpg_gene_edges = list(set(cpg_gene_edges))
                print(f"Extracted {len(cpg_gene_edges)} CpG-gene edges.")
        except Exception as e:
            print("Error parsing manifest:", e)
    else:
        print("WARNING: Manifest not found.")
        
    # --- SAVE ---
    if len(gene_gene_edges) == 0:
        # Create some random fallback edges to avoid crashing PyG
        gene_gene_edges = [(0, 0)]
    if len(mirna_gene_edges) == 0:
        mirna_gene_edges = [(0, 0)]
    if len(cpg_gene_edges) == 0:
        cpg_gene_edges = [(0, 0)]

    edge_index_dict = {
        ('gene', 'interacts', 'gene'): torch.tensor(gene_gene_edges, dtype=torch.long).t(),
        ('mirna', 'targets', 'gene'): torch.tensor(mirna_gene_edges, dtype=torch.long).t(),
        ('cpg', 'regulates', 'gene'): torch.tensor(cpg_gene_edges, dtype=torch.long).t(),
    }
    
    # Thêm cạnh 2 chiều (Undirected)
    # gene_gene vốn là undirected trong STRING.
    # ta thêm các rev_edges cho heterogeneous message passing
    edge_index_dict[('gene', 'rev_interacts', 'gene')] = edge_index_dict[('gene', 'interacts', 'gene')].flip(0)
    edge_index_dict[('gene', 'rev_targets', 'mirna')] = edge_index_dict[('mirna', 'targets', 'gene')].flip(0)
    edge_index_dict[('gene', 'rev_regulates', 'cpg')] = edge_index_dict[('cpg', 'regulates', 'gene')].flip(0)

    if HeteroData is not None:
        data = HeteroData()
        for k, v in edge_index_dict.items():
            data[k].edge_index = v
            
        data['gene'].num_nodes = len(gene_cols)
        data['mirna'].num_nodes = len(mirna_cols)
        data['cpg'].num_nodes = len(cpg_cols)
        
        torch.save(data, save_path)
        print(f"Saved PyG HeteroData to {save_path}")
    else:
        # Fallback dictionary
        meta = {
            "edge_index_dict": edge_index_dict,
            "num_nodes_dict": {
                'gene': len(gene_cols),
                'mirna': len(mirna_cols),
                'cpg': len(cpg_cols)
            }
        }
        torch.save(meta, save_path)
        print(f"Saved Edge Metadata to {save_path}")

if __name__ == "__main__":
    build_hetero_graph()
