import os
from ultralytics import YOLO

if __name__ == '__main__':
    print("="*60)
    print("🔥 LÒ LUYỆN ĐAN MEGA: 4 CẤU HÌNH x 5 FOLDS = 20 MẺ")
    print("="*60)

    # 4 Cấu hình lõi (Không liên quan NMS)
    configs = [
        {"name": "RAW_NoDA", "use_clahe": False, "use_da": False},
        {"name": "RAW_DA", "use_clahe": False, "use_da": True},
        {"name": "CLAHE_NoDA", "use_clahe": True, "use_da": False},
        {"name": "CLAHE_DA", "use_clahe": True, "use_da": True}
    ]

# --- BƯỚC 1: TỰ ĐỘNG MAP ĐƯỜNG DẪN CHO TẬP CLAHE (ĐÃ UPDATE DATA_SETTINGS) ---
    SETTINGS_DIR = "data_settings"
    
    for fold in range(1, 6):
        # 1.1: Tạo file .txt cho CLAHE
        for split in ['train', 'val']:
            raw_txt = f"{SETTINGS_DIR}/fold_{fold}_{split}.txt"
            clahe_txt = f"{SETTINGS_DIR}/fold_{fold}_{split}_clahe.txt"
            
            if os.path.exists(raw_txt) and not os.path.exists(clahe_txt):
                with open(raw_txt, 'r') as f:
                    lines = f.read()
                # Tráo thư mục ảnh gốc sang ảnh đã lọc sương CLAHE
                lines = lines.replace('dawn_yolo_dataset', 'dawn_clahe_dataset')
                with open(clahe_txt, 'w') as f:
                    f.write(lines)

        # 1.2: Tạo file .yaml cho CLAHE
        raw_yaml = f"{SETTINGS_DIR}/data_fold_{fold}.yaml"
        clahe_yaml = f"{SETTINGS_DIR}/data_fold_{fold}_clahe.yaml"
        
        if os.path.exists(raw_yaml) and not os.path.exists(clahe_yaml):
            with open(raw_yaml, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace('dawn_yolo_dataset', 'dawn_clahe_dataset')
            # Trỏ đường dẫn bên trong yaml sang file txt của CLAHE
            content = content.replace(f'fold_{fold}_train.txt', f'fold_{fold}_train_clahe.txt')
            content = content.replace(f'fold_{fold}_val.txt', f'fold_{fold}_val_clahe.txt')
            with open(clahe_yaml, 'w', encoding='utf-8') as f:
                f.write(content)
    # --- BƯỚC 2: CẮM MÁY TRAIN 20 LẦN ---
    for fold in range(1, 6):
        for cfg in configs:
            run_name = f"KFOLD_{cfg['name']}_Fold_{fold}"
            weight_path = f"runs/detect/{run_name}/weights/best.pt"
            
            # Tính năng chống sập: Bỏ qua nếu mẻ này đã train thành công
            if os.path.exists(weight_path):
                print(f"⏩ Đã có sẵn {run_name}, bỏ qua...")
                continue

            yaml_file = f"data_fold_{fold}_clahe.yaml" if cfg['use_clahe'] else f"data_fold_{fold}.yaml"
            
            if cfg['use_da']:
                da_args = {'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'mosaic': 1.0}
            else:
                da_args = {'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'mosaic': 0.0, 'mixup': 0.0, 'fliplr': 0.0}

            print(f"\n🚀 ĐANG TRAIN: {run_name}")
            model = YOLO('yolov8n.pt')
            model.train(
                data=yaml_file,
                epochs=300,
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

    print("\n✅ ĐÃ HOÀN THÀNH 20 MẺ TRAINING K-FOLD!")