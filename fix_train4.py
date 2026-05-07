import glob
import re

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()

    old_block_pattern = r'if args\.runs > 1:.*?else:\s+train\(args\)'
    
    new_block = '''if args.runs > 1:
        import numpy as np
        metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        base_seed = args.seed
        for i in range(args.runs):
            print(f"\\n{'='*50}\\nRUN {i+1}/{args.runs}\\n{'='*50}")
            args.seed = base_seed + i
            res = train(args)
            if res:
                for k in metrics:
                    if k in res: metrics[k].append(res[k])
        
        print(f"\\n{'='*50}\\nFINAL RESULTS OVER {args.runs} RUNS\\n{'='*50}")
        for k, v in metrics.items():
            if v:
                print(f"{k.capitalize()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    else:
        train(args)'''

    content = re.sub(old_block_pattern, new_block, content, flags=re.DOTALL)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
