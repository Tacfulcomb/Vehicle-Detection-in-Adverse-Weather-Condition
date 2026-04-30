import os
import shutil
import cv2
import glob
from ultralytics import YOLO

print("="*50)
print("CHẠY ZERO-SHOT FULL OPTION (CÓ MAP NHÃN COCO)")
print("="*50)

# ==========================================
# CẤU HÌNH TỔ HỢP TEST CASE 
# ==========================================
USE_PREPROCESS = True
USE_NMS = True          


RAW_VAL_LABEL_DIR = "dawn_yolo_dataset/labels/val"
RAW_IMG_DIR = "dawn_yolo_dataset/images/val"

if USE_PREPROCESS:
    ZSHOT_DIR = "dawn_zeroshot_clahe"
    print(" CHẾ ĐỘ: PREPROCESS (Dùng ảnh lọc CLAHE)")
else:
    ZSHOT_DIR = "dawn_zeroshot_raw"
    print(" CHẾ ĐỘ: RAW (Dùng ảnh gốc sương mù)")

ZSHOT_LABEL_DIR = f"{ZSHOT_DIR}/labels/val"
ZSHOT_IMG_DIR = f"{ZSHOT_DIR}/images/val"
YAML_FILE = f"{ZSHOT_DIR}.yaml"

LABEL_MAP = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '5', '5': '7'}

def apply_clahe(img_path, save_path):
    """Hàm áp dụng CLAHE lên ảnh"""
    img = cv2.imread(img_path)
    if img is None: return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    final_img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, final_img)

def prepare_dataset():
    if os.path.exists(ZSHOT_DIR):
        print(f" Thư mục {ZSHOT_DIR} đã có sẵn. Bỏ qua bước tạo data...")
        return YAML_FILE

    print(f" Đang tạo tập Zero-shot tại '{ZSHOT_DIR}'...")
    os.makedirs(ZSHOT_LABEL_DIR, exist_ok=True)
    os.makedirs(ZSHOT_IMG_DIR, exist_ok=True)
    
    # 1. Xử lý Ảnh (Có hoặc Không có CLAHE)
    img_files = os.listdir(RAW_IMG_DIR)
    for i, img_name in enumerate(img_files):
        src_img = os.path.join(RAW_IMG_DIR, img_name)
        dst_img = os.path.join(ZSHOT_IMG_DIR, img_name)
        if USE_PREPROCESS:
            apply_clahe(src_img, dst_img) # Áp dụng CLAHE
        else:
            shutil.copy(src_img, dst_img) # Copy nguyên bản
            
        if i % 100 == 0: print(f"   Đã xử lý ảnh {i}/{len(img_files)}...")

    # 2. Xử lý Nhãn (Khớp chuẩn COCO)
    for txt_file in os.listdir(RAW_VAL_LABEL_DIR):
        if not txt_file.endswith('.txt'): continue
        with open(os.path.join(RAW_VAL_LABEL_DIR, txt_file), 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] in LABEL_MAP:
                parts[0] = LABEL_MAP[parts[0]]
                new_lines.append(" ".join(parts) + "\n")
        with open(os.path.join(ZSHOT_LABEL_DIR, txt_file), 'w') as f:
            f.writelines(new_lines)
            
    # 3. Tạo file YAML
    yaml_content = f"""
path: {os.path.abspath(ZSHOT_DIR)}
train: images/val
val: images/val
names: {{0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck}}
"""
    with open(YAML_FILE, "w") as f:
        f.write(yaml_content.strip())
    print("✅ Hoàn tất chuẩn bị Dataset!")
    return YAML_FILE

# ----------------- THỰC THI -----------------
# 1. Data Preparation
current_yaml = prepare_dataset()

# 2. Setup NMS Parameters
if USE_NMS:
    print(" NMS: BẬT ")
    val_args = {'iou': 0.5, 'conf': 0.6}
else:
    print(" NMS: TẮT ")
    val_args = {'iou': 1.0, 'conf': 0.001}

# 3. Validate
print("\n Đang chạy Model Default (yolov8n.pt)...")
model = YOLO('yolov8n.pt') 

metrics = model.val(
    data=current_yaml, 
    classes=[0, 1, 2, 3, 5, 7], 
    **val_args
)

print(f"\n KẾT QUẢ mAP50: {metrics.box.map50 * 100:.2f}%")