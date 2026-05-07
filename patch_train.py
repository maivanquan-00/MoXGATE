import glob
import re
import os

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '--runs' in content and 'FINAL RESULTS' in content:
        print(f"Skipped {f} (already patched)")
        continue

    # 1. ADD argument --runs
    content = re.sub(
        r'(parser = argparse.ArgumentParser[^\n]*)',
        r'\1\n    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy tính trung bình")',
        content
    )

    # 2. Make train() return test_metrics:
    # We will just insert `return test_metrics` before `def parse_args():`
    # Warning, some files might have multiple line spacing, let's just find `def parse_args():`
    content = re.sub(
        r'\n(\s*def parse_args\(\):)',
        r'\n    return test_metrics\n\1',
        content
    )
    
    # 3. Modify the bottom __main__ code
    new_main = """if __name__ == "__main__":
    args = parse_args()
    if args.runs > 1:
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
        train(args)
"""
    content = re.sub(r'if __name__ == "__main__":.*', new_main, content, flags=re.DOTALL)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Patched {f}")
