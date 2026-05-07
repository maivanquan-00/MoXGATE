import glob
import re

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove any existing misplaced "    return test_metrics" 
    # anywhere before def parse_args():
    content = re.sub(r'^[ \t]*return test_metrics\s*$', '', content, flags=re.MULTILINE)
    
    # We want to put `return test_metrics` at the very end of `def train...`
    # Let's find the last line that belongs to train().
    # It usually ends with `json.dump(...)` or `print(Final Weights...)` or `plot_confusion_matrix(...)`
    # Let's match the block between `def train` and `def parse_args`
    match = re.search(r'(def train\(.*?\):.*?)(?=\n[# \-]*\n*def parse_args\(\):)', content, flags=re.DOTALL)
    if match:
        train_body = match.group(1)
        # remove trailing whitespaces/newlines from train_body
        train_body_stripped = train_body.rstrip()
        # count the indentation of the last line of train_body to know what space to use
        last_line = train_body_stripped.split('\n')[-1]
        indent = ' ' * (len(last_line) - len(last_line.lstrip()))
        if len(indent) == 0:
            indent = '    ' # Default
        
        # append properly
        new_train_body = train_body_stripped + '\n' + indent + 'return test_metrics\n\n'
        content = content.replace(train_body, new_train_body)
    else:
        # Fallback if no exact match (like train_balanced.py)
        pass

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Fixed {f}")
