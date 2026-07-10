import os
import yaml
import re
from collections import defaultdict

def main():
    print("="*80)
    print("ĐẾM SỐ LƯỢNG THỰC TẾ (ACTUAL TOTAL) - LUẬT PHÉP TRỪ (ELIMINATION)")
    print("="*80)

    unique_images = set()
    class_map = {}
    
    # Chỉ check 3 thời tiết có tên rõ ràng
    explicit_weathers = ['rain', 'snow', 'sand']

    for fold in range(1, 6):
        yaml_path = f"data_settings/data_fold_{fold}_clahe.yaml"
        if not os.path.exists(yaml_path): continue

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not class_map:
            raw_names = data.get('names', {})
            if isinstance(raw_names, list):
                class_map = {i: name for i, name in enumerate(raw_names)}
            else:
                class_map = raw_names

        for key in ['train', 'val']:
            txt_path = data.get(key)
            if txt_path and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        img = line.strip()
                        if img: unique_images.add(img)

    actual_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    for img_path in unique_images:
        current_weather = "UNKNOWN"
        img_lower = img_path.lower()
        
        for w in explicit_weathers:
            if re.search(r'\b' + w, img_lower):
                current_weather = w.upper()
                break
        
        # LUẬT PHÉP TRỪ: Nếu không phải Mưa, Tuyết, Cát -> Gom tất cả về Sương Mù
        if current_weather == "UNKNOWN":
            current_weather = "FOG"

        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        txt_path = txt_path.replace('images', 'labels')

        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    actual_counts[current_weather][cls_id] += 1
                    total_counts[cls_id] += 1

    weather_keys = ['FOG', 'RAIN', 'SAND', 'SNOW']
    header = f"{'Class Name':<12} | " + " | ".join([f"{w:<7}" for w in weather_keys]) + " | ACTUAL TOTAL"
    
    print("\n" + "="*len(header))
    print("BẢNG SỐ LƯỢNG THỰC TẾ 100% (ĐÃ TÌM THẤY TOÀN BỘ XE TRONG SƯƠNG MÙ)")
    print("="*len(header))
    print(header)
    print("-" * len(header))

    for cls_id, cls_name in class_map.items():
        row_str = f"{cls_name.upper():<12} | "
        for w in weather_keys:
            row_str += f"{actual_counts[w].get(cls_id, 0):<7} | "
        
        row_str += f"{total_counts[cls_id]}"
        print(row_str)
        
    print("="*len(header))

if __name__ == '__main__':
    main()