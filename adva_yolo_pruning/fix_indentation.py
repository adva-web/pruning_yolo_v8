#!/usr/bin/env python3
"""Fix indentation in run_4_methods_comparison_blocks_1357_coco.py"""

file_path = "adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix line 1605 (0-indexed: 1604)
if len(lines) > 1604:
    line = lines[1604]
    # Replace 20 spaces with 16 spaces at the start
    if line.startswith('                    instances'):
        lines[1604] = '                instances = getattr(test_results.metrics, \'instances\', 0)\n'
        print(f"Fixed line 1605")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Done!")

