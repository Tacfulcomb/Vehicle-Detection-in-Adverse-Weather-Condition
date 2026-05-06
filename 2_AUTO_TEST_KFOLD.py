import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# HÀM BỔ TRỢ: TỰ ĐỘNG LƯU LOG RA FILE CSV
# ==========================================
def log_to_csv(test_name, map_score, log_file="K_Fold_Results_Log.csv"):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Thời gian", "Tên Test Case", "mAP@50 (%)"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, test_name, round(map_score, 2)])

if __name__ == '__main__':
    print("="*60)
    print("🔍 BẮT ĐẦU CHẤM ĐIỂM 5-FOLD & LƯU TỰ ĐỘNG RA CSV")
    print("="*60)

    configs = [
        {"name": "RAW_NoDA", "use_clahe": False},
        {"name": "RAW_DA", "use_clahe": False},
        {"name": "CLAHE_NoDA", "use_clahe": True},
        {"name": "CLAHE_DA", "use_clahe": True}
    ]

    total_scores = {}

    for cfg in configs:
        base_name = cfg['name']
        total_scores[f"{base_name}_NMS_ON"] = 0.0
        total_scores[f"{base_name}_NMS_OFF"] = 0.0

        for fold in range(1, 6):
            run_name = f"KFOLD_{base_name}_Fold_{fold}"
            weight_path = f"runs/detect/{run_name}/weights/best.pt"
            yaml_file = f"data_settings/data_fold_{fold}_clahe.yaml" if cfg['use_clahe'] else f"data_settings/data_fold_{fold}.yaml"

            if not os.path.exists(weight_path):
                print(f"❌ Bỏ qua {run_name} vì chưa train xong!")
                continue

            print(f"\n--- Đang Test: {run_name} ---")
            model = YOLO(weight_path)

            # Test NMS ON
            metrics_on = model.val(data=yaml_file,plots = False,saves = False, split='val', iou=0.5, conf=0.25, verbose=False, workers=0, name=f"TEST_{run_name}_ON") 
            map_on = metrics_on.box.map50 * 100
            total_scores[f"{base_name}_NMS_ON"] += map_on
            log_to_csv(f"{run_name}_NMS_ON", map_on) # Ghi log dòng này

            # Test NMS OFF
            metrics_off = model.val(data=yaml_file,plots = False,saves = False, split='val', iou=1.0, conf=0.001, verbose=False, workers=0, name=f"TEST_{run_name}_OFF") 
            map_off = metrics_off.box.map50 * 100
            total_scores[f"{base_name}_NMS_OFF"] += map_off
            log_to_csv(f"{run_name}_NMS_OFF", map_off) # Ghi log dòng này

    # In Bảng Tổng và lưu điểm Trung bình cuối cùng
    print("\n" + "="*60)
    print("🏆 BẢNG KẾT QUẢ TRUNG BÌNH 5-FOLD 🏆")
    print("="*60)
    for test_case, total_map in total_scores.items():
        avg_map = total_map / 5.0
        print(f"{test_case: <25} | mAP@50 Trung bình: {avg_map:.2f}%")
        # Ghi luôn cái điểm trung bình này vào CSV cho dễ lọc
        log_to_csv(f"TRUNG_BINH_CHUNG_{test_case}", avg_map)
    print("="*60)
    print("✅ Đã xuất toàn bộ dữ liệu ra file: K_Fold_Results_Log.csv")