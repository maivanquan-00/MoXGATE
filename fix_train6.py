import glob
import re

for f in glob.glob('train*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()

    # The broken injects:
    # `    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy tính trung bình")`
    broken_str = r'\n    parser\.add_argument\("--runs", type=int, default=1, help="Số lần chạy tính trung bình"\)'
    content = re.sub(broken_str, '', content)

    # Sometimes there is `    parser.add_argument('--runs', type=int, default=1)` from earlier tries? No, I ran it. Let's just remove anything that matches `--runs`:
    # Actually just string literal replace is safer:
    
    content = content.replace('    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy tính trung bình")', '')
    content = content.replace("    parser.add_argument('--runs', type=int, default=1)", "")

    # Now let's inject it cleanly before return parser.parse_args()
    
    clean_injection = '''
    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy lấy trung bình trên Colab")
    return parser.parse_args()'''
    
    content = content.replace('return parser.parse_args()', clean_injection.strip())

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
