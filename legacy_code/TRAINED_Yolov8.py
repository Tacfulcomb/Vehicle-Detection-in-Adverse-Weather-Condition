import os
import cv2
import shutil
import glob
from ultralytics import YOLO

USE_PREPROCESS = True  
USE_DA = True          

EPOCHS = 300            

# DIR DECLARATION
RAW_DIR = "dawn_yolo_dataset"
CLAHE_DIR = "dawn_clahe_dataset"
RAW_YAML = "data_raw.yaml"
CLAHE_YAML = "data_clahe.yaml"

def apply_clahe(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None: return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    final_img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, final_img)

def prepare_clahe_data():
    if os.path.exists(CLAHE_DIR):
        return CLAHE_YAML
    print(" Đang tạo tập ảnh CLAHE... (Chỉ chạy 1 lần duy nhất)")
    shutil.copytree(RAW_DIR, CLAHE_DIR)
    images = glob.glob(f"{CLAHE_DIR}/images/**/*.jpg", recursive=True) + glob.glob(f"{CLAHE_DIR}/images/**/*.png", recursive=True)
    for i, img in enumerate(images):
        apply_clahe(img, img)
    
    yaml_content = f"path: {os.path.abspath(CLAHE_DIR)}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n  4: bus\n  5: truck"
    with open(CLAHE_YAML, 'w') as f: f.write(yaml_content)
    return CLAHE_YAML

# MAIN
if __name__ == '__main__':
    # 1. Chọn Data
    if USE_PREPROCESS:
        print("\n CHẾ ĐỘ DATA: PREPROCESS (CLAHE)")
        current_yaml = prepare_clahe_data()
    else:
        print("\n CHẾ ĐỘ DATA: RAW (Gốc)")
        current_yaml = RAW_YAML

    model = YOLO('yolov8n.pt')

    # 2. Chọn Data Augmentation
    if USE_DA:
        print(" DATA AUGMENTATION: BẬT")
        da_args = {'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'mosaic': 1.0}
    else:
        print(" DATA AUGMENTATION: TẮT")
        da_args = {'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'mosaic': 0.0, 'mixup': 0.0, 'fliplr': 0.0}

    run_name = f"{'CLAHE' if USE_PREPROCESS else 'RAW'}_DA_{USE_DA}"

    print(f"\n BẮT ĐẦU TRAIN MODEL: {run_name} ({EPOCHS} Epochs)")
    
    # 3. Tiến hành Train
    model.train(
        data=current_yaml,
        epochs=EPOCHS,
        patience=50,
        batch=4,
        imgsz=640,
        lr0=0.01,
        weight_decay=0.0005,
        device=0, 
        name=run_name,
        workers=0, 
        **da_args
    )
    
    print(f"\n ĐÃ TRAIN XONG MẺ: {run_name}")
    print(f" File best.pt đã được lưu tại: runs/detect/{run_name}/weights/best.pt")