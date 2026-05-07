import glob

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    cleaned_lines = []
    # 1. Remove all old "return test_metrics" lines entirely
    for line in lines:
        if 'return test_metrics' not in line:
            cleaned_lines.append(line)
            
    # 2. Insert `    return test_metrics\n` right before the comment block indicating ARGUMENT PARSER,
    # or just before `def parse_args():` if there's no comment block.
    # We will look for def parse_args(), then walk backwards and insert before any module level comments.
    insert_idx = -1
    for i, line in enumerate(cleaned_lines):
        if line.strip().startswith('def parse_args():'):
            # Walk backwards backwards to skip blank lines and comments
            for j in range(i-1, -1, -1):
                if cleaned_lines[j].strip() and not cleaned_lines[j].strip().startswith('#'):
                    # Found last line of train()
                    insert_idx = j
                    break
            break
            
    if insert_idx != -1:
        # Check standard indent: usually 4 spaces.
        cleaned_lines.insert(insert_idx + 1, '    return test_metrics\n')
        
    with open(f, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)
    
    print(f"Final fix {f}")
