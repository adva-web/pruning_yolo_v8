#!/usr/bin/env python3
import sys

file_path = "adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix line 1605 (0-indexed: 1604) - reduce indentation from 20 to 16 spaces
if len(lines) > 1604:
    line = lines[1604]
    if 'instances = getattr' in line and line.startswith('                    '):
        # Replace 20 spaces with 16 spaces
        new_line = '                ' + line.lstrip()
        lines[1604] = new_line
        print(f"Fixed line 1605: {repr(line[:30])} -> {repr(new_line[:30])}")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Done fixing indentation!")

