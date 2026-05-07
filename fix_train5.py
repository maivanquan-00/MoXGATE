import glob

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()

    # In some places it is print(f"\n{'='*50}\nRUN etc...") with raw newlines.
    bad_part = """print(f"
{'='*50}
RUN {i+1}/{args.runs}
{'='*50}")"""
    good_part = "print(f\"\\n{'='*50}\\nRUN {i+1}/{args.runs}\\n{'='*50}\")"
    content = content.replace(bad_part, good_part)
    
    bad_part2 = """print(f"
{'='*50}
FINAL RESULTS OVER {args.runs} RUNS
{'='*50}")"""
    good_part2 = "print(f\"\\n{'='*50}\\nFINAL RESULTS OVER {args.runs} RUNS\\n{'='*50}\")"
    content = content.replace(bad_part2, good_part2)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
