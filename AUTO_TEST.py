import os
from ultralytics import YOLO

if __name__ == '__main__':
    print("="*60)
    print("BAT DAU CHAM DIEM 8 TEST CASES (KHONG TRAIN)")
    print("="*60)

    RAW_YAML = "data_raw.yaml"
    CLAHE_YAML = "data_clahe.yaml"

    trained_models = [
        {"name": "RAW + No DA",    "weight": "runs/detect/RAW_DA_False/weights/best.pt", "yaml": RAW_YAML},
        {"name": "RAW + DA",       "weight": "runs/detect/RAW_DA_True/weights/best.pt",  "yaml": RAW_YAML},
        {"name": "CLAHE + No DA",  "weight": "runs/detect/CLAHE_DA_False/weights/best.pt", "yaml": CLAHE_YAML},
        {"name": "CLAHE + DA",     "weight": "runs/detect/CLAHE_DA_True/weights/best.pt",  "yaml": CLAHE_YAML}
    ]

    results_summary = []

    for model_info in trained_models:
        weight_path = model_info["weight"]
        yaml_path = model_info["yaml"]
        model_name = model_info["name"]

        if not os.path.exists(weight_path):
            print(f"[LOI] Khong tim thay file weight: {weight_path}. Vui long kiem tra lai!")
            continue

        print(f"\n--- DANG TEST MODEL: {model_name} ---")
        
        model = YOLO(weight_path)
        safe_folder_name = model_name.replace(" + ", "_").replace(" ", "_")

        # TEST CASE 1: BẬT NMS 
        print("-> Dang kiem tra NMS = ON...")
        val_name_on = f"TEST_{safe_folder_name}_NMS_ON"
        metrics_on = model.val(data=yaml_path, split='val', iou=0.6, conf=0.25, verbose=False, workers=0,
                               name = val_name_on) 
        map_on = metrics_on.box.map50 * 100

        # TEST CASE 2: TẮT NMS 
        print("-> Dang kiem tra NMS = OFF...")
        val_name_off = f"TEST_{safe_folder_name}_NMS_OFF"
        metrics_off = model.val(data=yaml_path, split='val', iou=1.0, conf=0.001, verbose=False, workers=0,
                                name=val_name_off) 
        map_off = metrics_off.box.map50 * 100

        # Luu ket qua
        results_summary.append((f"{model_name} + NMS ON", map_on))
        results_summary.append((f"{model_name} + NMS OFF", map_off))

    print("\n" + "="*60)
    print("BANG TONG KET 8 TEST CASES MA TRAN")
    print("="*60)
    for test_name, map_score in results_summary:
        print(f"{test_name: <30} | mAP@50: {map_score:.2f}%")
    print("="*60)