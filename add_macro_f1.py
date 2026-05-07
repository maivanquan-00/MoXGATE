import glob
import re

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '"macro_f1"' in content:
        continue # already updated
        
    # 1. Update evaluate dictionary
    # We look for "f1" or 'f1' inside the dict returned by evaluate
    content = re.sub(
        r'([\'"]f1[\'"]\s*:\s*f1_score\([^,]+,\s*[^,]+,\s*average=[\'"]weighted[\'"],\s*zero_division=0\),?)',
        r'"weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),\n        "macro_f1":    f1_score(all_targets, all_preds, average="macro", zero_division=0),',
        content
    )
    
    # 2. Update multi-run metric dict
    content = content.replace(
        "metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}",
        "metrics = {'accuracy': [], 'weighted_f1': [], 'macro_f1': [], 'precision': [], 'recall': []}"
    )
    
    # 3. Update printed test results
    content = re.sub(
        r'print\(f"[\\n]*Test F1\s*: \{test_metrics\[[\'"]f1[\'"]\]:\.4f\}"\)',
        r'print(f"Test Weighted F1: {test_metrics[\'weighted_f1\']:.4f}")\n    print(f"Test Macro F1   : {test_metrics[\'macro_f1\']:.4f}")',
        content
    )
    
    # 4. Update Val printing
    content = re.sub(
        r'Val F1: \{val_metrics\[[\'"]f1[\'"]\]:\.4f\}',
        r'Val W-F1: {val_metrics[\'weighted_f1\']:.4f} | Val M-F1: {val_metrics[\'macro_f1\']:.4f}',
        content
    )
    
    # 5. Replace any remaining metric references of f1 -> weighted_f1
    content = re.sub(r'val_metrics\[[\'"]f1[\'"]\]', "val_metrics['weighted_f1']", content)
    content = re.sub(r'test_metrics\[[\'"]f1[\'"]\]', "test_metrics['weighted_f1']", content)
    
    # 6. Make multi-run print look nice
    old_print_loop = r'print\(f"\{k\.capitalize\(\)\}: \{np\.mean\(v\):\.4f\} ± \{np\.std\(v\):\.4f\}"\)'
    new_print_loop = r'name_map = {"accuracy": "Accuracy", "weighted_f1": "Weighted F1", "macro_f1": "Macro F1", "precision": "Precision", "recall": "Recall"}\n                print(f"{name_map.get(k, k.capitalize())}: {np.mean(v):.4f} ± {np.std(v):.4f}")'
    content = re.sub(old_print_loop, new_print_loop, content)

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
        
    print(f"Added Macro/Weighted F1 to {f}")
