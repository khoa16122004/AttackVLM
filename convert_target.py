import os
anno_path = "annotations.txt"

with open(anno_path, 'r') as f:
    lines = [line.strip().split("\t")[1] for line in f.readlines()]
    
with open("gt_annotations.txt", 'w') as f:
    for line in lines:
        f.write(line + "\n")