import os
import yaml
from collections import defaultdict

def main():
    print("="*80)
    print("DAWN DATASET: TRUY TÌM NHỮNG THỜI TIẾT BỊ ẨN GIẤU")
    print("="*80)

    weather_counts = defaultdict(lambda: defaultdict(int))
    grand_total_counts = defaultdict(int)
    class_map = {}
    
    # Tập hợp để lưu xem có những thời tiết nào tự động được tìm thấy
    found_weathers = set()

    for fold in range(1, 6):
        fold_yaml_path = f"data_settings/data_fold_{fold}_clahe.yaml"
        if not os.path.exists(fold_yaml_path): continue

        with open(fold_yaml_path, 'r', encoding='utf-8') as f:
            fold_data = yaml.safe_load(f)
            
        if not class_map:
            raw_names = fold_data.get('names', {})
            if isinstance(raw_names, list):
                class_map = {i: name for i, name in enumerate(raw_names)}
            else:
                class_map = raw_names

        val_txt_path = fold_data.get('val')
        if not os.path.exists(val_txt_path): continue

        with open(val_txt_path, 'r', encoding='utf-8') as f:
            val_images = [line.strip() for line in f.readlines() if line.strip()]

        for img_path in val_images:
            img_name = os.path.basename(img_path).lower()
            
            # TỰ ĐỘNG BÓC TÁCH TÊN THỜI TIẾT TỪ TÊN FILE
            # Ví dụ: "mist_001.jpg" -> Cắt tại dấu "_" -> Lấy chữ "mist"
            if '_' in img_name:
                current_weather = img_name.split('_')[0].upper()
            else:
                current_weather = "UNKNOWN"
                
            found_weathers.add(current_weather)

            txt_path = img_path.rsplit('.', 1)[0] + '.txt'
            txt_path = txt_path.replace('images', 'labels')
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as label_file:
                    for line in label_file:
                        cls_id = int(line.strip().split()[0])
                        weather_counts[current_weather][cls_id] += 1
                        grand_total_counts[cls_id] += 1

    # Sắp xếp danh sách thời tiết tìm được
    weather_list = sorted(list(found_weathers))

    print(f"\n[PHÁT HIỆN] Các tiền tố thời tiết đang có trong Dataset của ông:")
    print(", ".join(weather_list))

    print("\n" + "="*80)
    print("BẢNG PHÂN BỐ DỮ LIỆU CHÍNH XÁC 100%")
    print("="*80)
    
    # Tạo Header linh hoạt dựa trên số lượng thời tiết tìm được
    header = f"{'Class Name':<12} | " + " | ".join([f"{w:<6}" for w in weather_list]) + " | TOTAL"
    print(header)
    print("-" * len(header))
    
    for cls_id, cls_name in class_map.items():
        row_str = f"{cls_name.upper():<12} | "
        
        for w in weather_list:
            count = weather_counts[w].get(cls_id, 0)
            row_str += f"{count:<6} | "
            
        total = grand_total_counts.get(cls_id, 0)
        row_str += f"{total}"
        print(row_str)

if __name__ == '__main__':
    main()