import os

annotation_file_path = "annotations.txt" 
target_image_dir = "target_image/samples"

with open(annotation_file_path, 'r') as f:
    lines = [line.strip().split()[0] for line in f.readlines()]  # Sửa lỗi split()

for i, img_name in enumerate(sorted(lines)):
    old_path = os.path.join(target_image_dir, img_name)
    print(img_name)
    continue
    new_name = lines[i]  # Cập nhật tên từ annotation
    new_path = os.path.join(target_image_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
    else:
        print(f"File not found: {old_path}")
    
