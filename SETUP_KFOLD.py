import os
import glob
import random

print("="*60)
print("BẮT ĐẦU CHIA DATASET THEO CHUẨN 5-FOLD CROSS-VALIDATION (BẢN GỌN GÀNG)")
print("="*60)

K_FOLDS = 5
DATASET_DIR = "dawn_yolo_dataset" 
SETTINGS_DIR = "data_settings" 

# Tạo thư mục data_settings nếu nó chưa tồn tại
os.makedirs(SETTINGS_DIR, exist_ok=True)

# 1. Gom tất cả đường dẫn ảnh gốc
all_images = []
for ext in ['*.jpg', '*.png', '*.jpeg']:
    all_images.extend(glob.glob(f"{DATASET_DIR}/images/**/{ext}", recursive=True))

all_images = [os.path.abspath(img).replace('\\', '/') for img in all_images]

# Xáo trộn danh sách ảnh (Giữ nguyên seed 42 để không lệch data cũ)
random.seed(42) 
random.shuffle(all_images)

total_images = len(all_images)
fold_size = total_images // K_FOLDS

print(f"Tổng số ảnh tìm thấy: {total_images}")

# 2. Cắt bánh và tạo YAML
for fold in range(K_FOLDS):
    start_idx = fold * fold_size
    end_idx = start_idx + fold_size if fold < K_FOLDS - 1 else total_images
    
    val_images = all_images[start_idx:end_idx]
    train_images = all_images[:start_idx] + all_images[end_idx:]
    
    # Ép nó lưu file .txt vào trong thư mục data_settings
    train_txt = f"{SETTINGS_DIR}/fold_{fold + 1}_train.txt"
    val_txt = f"{SETTINGS_DIR}/fold_{fold + 1}_val.txt"
    
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_images))
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_images))
        
    yaml_name = f"{SETTINGS_DIR}/data_fold_{fold + 1}.yaml"
    
    yaml_content = f"""path: {os.path.abspath(DATASET_DIR).replace('\\', '/')}
train: {os.path.abspath(train_txt).replace('\\', '/')}
val: {os.path.abspath(val_txt).replace('\\', '/')}

# Số class và Tên class chuẩn
nc: 6
names: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
"""
    with open(yaml_name, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"✅ Đã tạo xong Fold {fold + 1} và lưu vào thư mục '{SETTINGS_DIR}'")

print("="*60)
print("ĐÃ CHIA XONG! Không còn rác ở thư mục gốc nữa nhé.")