import glob

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()

    # Pyton f-strings cannot have backslashes inside {}.
    # My regex replaced used `\'` inside `{...}`, meaning `{val_metrics[\'weighted_f1\']}`.
    # Needs to be just `'` or `"`
    
    content = content.replace("val_metrics[\\'weighted_f1\\']", "val_metrics['weighted_f1']")
    content = content.replace("val_metrics[\\'macro_f1\\']", "val_metrics['macro_f1']")
    content = content.replace("test_metrics[\\'weighted_f1\\']", "test_metrics['weighted_f1']")
    content = content.replace("test_metrics[\\'macro_f1\\']", "test_metrics['macro_f1']")

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
