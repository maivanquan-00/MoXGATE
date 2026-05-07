import glob
import re

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find where `def train` is defined, and where `def parse_args` is defined
    train_end_idx = -1
    for i in range(len(lines)):
        if lines[i].startswith('def parse_args'):
            # The line before parse_args, but we need to skip any comments and blank lines backwards
            for j in range(i-1, -1, -1):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    # that's the last line of train!
                    train_end_idx = j
                    break
            break
            
    if train_end_idx != -1:
        # First, ensure we remove any existing "return test_metrics" to avoid duplicates
        for i in range(len(lines)):
            if 'return test_metrics' in lines[i]:
                lines[i] = '\n'
                
        # Now insert right after `train_end_idx`
        # get indent
        last_line = lines[train_end_idx]
        indent_len = len(last_line) - len(last_line.lstrip())
        indent = ' ' * indent_len if indent_len > 0 else '    '
        
        lines.insert(train_end_idx + 1, f'{indent}return test_metrics\n')
        
    with open(f, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print(f"Corrected {f}")
