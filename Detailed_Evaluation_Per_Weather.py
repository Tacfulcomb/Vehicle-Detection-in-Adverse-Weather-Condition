import os
import csv
import yaml
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import math
import re

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_boxes_from_txt(txt_path, img_w, img_h):
    boxes = []
    if not os.path.exists(txt_path): return boxes
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            x1 = int((x_center - w/2) * img_w)
            y1 = int((y_center - h/2) * img_h)
            x2 = int((x_center + w/2) * img_w)
            y2 = int((y_center + h/2) * img_h)
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

def save_image_grid(images, output_dir, prefix, cols=4, rows=4, img_size=640):
    if not images: return
    os.makedirs(output_dir, exist_ok=True)
    
    imgs_per_page = cols * rows
    total_pages = math.ceil(len(images) / imgs_per_page)
    
    for page in range(total_pages):
        page_imgs = images[page * imgs_per_page : (page + 1) * imgs_per_page]
        
        while len(page_imgs) < imgs_per_page:
            blank_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            page_imgs.append(blank_image)
            
        resized_imgs = [cv2.resize(img, (img_size, img_size)) for img in page_imgs]
        
        grid_rows = []
        for r in range(rows):
            row_imgs = resized_imgs[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_imgs))
            
        final_grid = np.vstack(grid_rows)
        save_path = os.path.join(output_dir, f"{prefix}_Error_Grid_Page_{page + 1}.jpg")
        cv2.imwrite(save_path, final_grid)

def count_instances_manually(img_paths, names):
    class_counts = {i: 0 for i in range(len(names))}
    for img_path in img_paths:
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        txt_path = txt_path.replace('images', 'labels')
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
    return class_counts

def filter_images_by_weather(image_list, weather):
    filtered_imgs = []
    explicit_weathers = ['rain', 'snow', 'sand']
    
    for img in image_list:
        img_lower = img.lower()
        if weather == 'fog':
            # Nếu có chữ fog, hoặc KHÔNG chứa rain, snow, sand thì gom vào Fog
            if re.search(r'\bfog', img_lower) or not any(re.search(r'\b' + w, img_lower) for w in explicit_weathers):
                filtered_imgs.append(img)
        else:
            if re.search(r'\b' + weather, img_lower):
                filtered_imgs.append(img)
                
    return filtered_imgs

def run_evaluation_for_model(base_model_name):
    print(f"\n[{base_model_name}] STARTING 5-FOLD EVALUATION AND GRID ERROR TRACKING")
    
    my_conf = 0.25
    my_iou = 0.6
    iou_threshold = 0.3 
    target_classes = ['bicycle', 'motorcycle']
    weathers = ['fog', 'rain', 'snow', 'sand']
    
    csv_file = f"Strict_5Fold_Metrics_{base_model_name}.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Weather", "Class", "Train_Instances", "Val_Instances", "Precision(%)", "Recall(%)", "mAP@50(%)", "mAP@50-95(%)"])

    error_base_dir = f"Error_Analysis_{base_model_name}_Grids"
    plots_base_dir = f"Plots_{base_model_name}"
    
    all_error_images = []

    for fold in range(1, 6):
        weight_path = f"runs/detect/KFOLD_{base_model_name}_Fold_{fold}/weights/best.pt"
        fold_yaml_path = f"data_settings/data_fold_{fold}_clahe.yaml"
        
        if not os.path.exists(weight_path) or not os.path.exists(fold_yaml_path): continue

        model = YOLO(weight_path)
        names = model.names
        target_ids = [k for k, v in names.items() if v.lower() in target_classes]

        with open(fold_yaml_path, 'r', encoding='utf-8') as f:
            fold_data = yaml.safe_load(f)
        
        train_txt_path = fold_data.get('train')
        val_txt_path = fold_data.get('val')
        
        if not val_txt_path or not os.path.exists(val_txt_path): continue
        if not train_txt_path or not os.path.exists(train_txt_path): continue

        with open(train_txt_path, 'r', encoding='utf-8') as f:
            all_train_images = [line.strip() for line in f.readlines() if line.strip()]

        with open(val_txt_path, 'r', encoding='utf-8') as f:
            all_val_images = [line.strip() for line in f.readlines() if line.strip()]

        for weather in weathers:
            # SỬ DỤNG HÀM LỌC MỚI ĐỂ BẮT GỌN MỌI FILE FOG ẨN DANH
            weather_train_imgs = filter_images_by_weather(all_train_images, weather)
            weather_val_imgs = filter_images_by_weather(all_val_images, weather)
            
            if not weather_val_imgs: continue
            
            train_counts = count_instances_manually(weather_train_imgs, names)
            val_counts = count_instances_manually(weather_val_imgs, names)

            temp_txt = os.path.abspath(f"temp_fold{fold}_{weather}_{base_model_name}.txt")
            with open(temp_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(weather_val_imgs))
                
            temp_yaml = f"temp_fold{fold}_{weather}_{base_model_name}.yaml"
            with open(temp_yaml, 'w', encoding='utf-8') as f:
                yaml.dump({
                    'path': fold_data.get('path', './'),
                    'train': temp_txt, 
                    'val': temp_txt,
                    'nc': len(names),
                    'names': names
                }, f)

            metrics = model.val(
                data=temp_yaml, split='val', conf=my_conf, iou=my_iou, plots=True,
                project=plots_base_dir, name=f"Fold_{fold}_{weather}", verbose=False, workers=0
            )
            
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for i, class_idx in enumerate(metrics.box.ap_class_index):
                    c_name = names[class_idx]
                    writer.writerow([
                        fold, weather.upper(), c_name, 
                        train_counts[class_idx], 
                        val_counts[class_idx],   
                        round(metrics.box.p[i]*100, 2), round(metrics.box.r[i]*100, 2), 
                        round(metrics.box.ap50[i]*100, 2), round(metrics.box.ap[i]*100, 2)
                    ])

            for img_path in weather_val_imgs:
                txt_path = img_path.rsplit('.', 1)[0] + '.txt'
                txt_path = txt_path.replace('images', 'labels')
                
                img = cv2.imread(img_path)
                if img is None: continue
                h, w = img.shape[:2]
                
                gt_boxes = get_boxes_from_txt(txt_path, w, h)
                results = model.predict(img_path, conf=my_conf, iou=my_iou, verbose=False)
                
                pred_boxes = []
                for box in results[0].boxes:
                    pred_boxes.append((int(box.cls[0]), int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), float(box.conf[0])))
                
                matched_gt = set()
                matched_pred = set()
                pred_boxes_sorted = sorted(enumerate(pred_boxes), key=lambda x: x[1][5], reverse=True)
                
                for p_idx, p_box in pred_boxes_sorted:
                    best_iou = 0
                    best_gt_idx = -1
                    for g_idx, g_box in enumerate(gt_boxes):
                        if g_idx not in matched_gt and p_box[0] == g_box[0]:
                            iou = calculate_iou(p_box[1:5], g_box[1:5])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                    if best_iou >= iou_threshold:
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(p_idx)
                
                is_target_error = False
                unmatched_gt_ids = [i for i in range(len(gt_boxes)) if i not in matched_gt]
                unmatched_pred_ids = [i for i in range(len(pred_boxes)) if i not in matched_pred]
                
                for g_idx in unmatched_gt_ids:
                    if gt_boxes[g_idx][0] in target_ids:
                        is_target_error = True
                        break
                if not is_target_error:
                    for p_idx in unmatched_pred_ids:
                        if pred_boxes[p_idx][0] in target_ids:
                            is_target_error = True
                            break
                
                if is_target_error:
                    for (cls, x1, y1, x2, y2) in gt_boxes:
                        if cls in target_ids:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"GT:{names[cls]}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for (cls, x1, y1, x2, y2, pconf) in pred_boxes:
                        if cls in target_ids:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img, f"PR:{names[cls]} {pconf:.2f}", (x1, min(h, y2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    banner_height = 40
                    cv2.rectangle(img, (0, 0), (w, banner_height), (0, 0, 0), -1)
                    cv2.putText(img, f"ENV: {weather.upper()} | FOLD: {fold}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    all_error_images.append(img)

            if os.path.exists(temp_txt): os.remove(temp_txt)
            if os.path.exists(temp_yaml): os.remove(temp_yaml)

    save_image_grid(all_error_images, error_base_dir, base_model_name, cols=4, rows=4, img_size=640)

def convert_and_aggregate():
    print("\nSTARTING DATA CONVERSION AND AGGREGATION...")
    raw_file = "Strict_5Fold_Metrics_RAW_DA.csv"
    clahe_file = "Strict_5Fold_Metrics_CLAHE_DA.csv"
    
    if not os.path.exists(raw_file) or not os.path.exists(clahe_file):
        print("Error: Missing CSV files. Please check the directory.")
        return

    for file in [raw_file, clahe_file]:
        df = pd.read_csv(file)
        df.to_excel(file.replace('.csv', '.xlsx'), index=False)

    df_raw = pd.read_csv(raw_file)
    df_clahe = pd.read_csv(clahe_file)
    df_raw['Model'] = 'RAW'
    df_clahe['Model'] = 'CLAHE'

    df_all = pd.concat([df_raw, df_clahe], ignore_index=True)
    
    agg_rules = {
        'Train_Instances': 'sum', 
        'Val_Instances': 'sum', 
        'Precision(%)': 'mean', 
        'Recall(%)': 'mean', 
        'mAP@50(%)': 'mean', 
        'mAP@50-95(%)': 'mean'
    }
    df_summary = df_all.groupby(['Model', 'Class', 'Weather']).agg(agg_rules).reset_index()
    
    p = df_summary['Precision(%)']
    r = df_summary['Recall(%)']
    df_summary['F1-Score(%)'] = np.where((p + r) > 0, 2 * (p * r) / (p + r), 0)

    pivot_ult = df_summary.pivot(
        index=['Class', 'Weather', 'Train_Instances', 'Val_Instances'], columns='Model', 
        values=['Precision(%)', 'Recall(%)', 'F1-Score(%)', 'mAP@50(%)', 'mAP@50-95(%)']
    )
    pivot_ult.columns = [f"{col[1]}_{col[0]}" for col in pivot_ult.columns]
    pivot_ult = pivot_ult.reset_index()

    pivot_ult.rename(columns={'Val_Instances': 'Total_Instances_Per_Weather'}, inplace=True)

    pivot_ult['Diff_Precision'] = pivot_ult['CLAHE_Precision(%)'] - pivot_ult['RAW_Precision(%)']
    pivot_ult['Diff_Recall'] = pivot_ult['CLAHE_Recall(%)'] - pivot_ult['RAW_Recall(%)']
    pivot_ult['Diff_F1'] = pivot_ult['CLAHE_F1-Score(%)'] - pivot_ult['RAW_F1-Score(%)']
    pivot_ult['Diff_mAP50'] = pivot_ult['CLAHE_mAP@50(%)'] - pivot_ult['RAW_mAP@50(%)']
    pivot_ult['Diff_mAP50-95'] = pivot_ult['CLAHE_mAP@50-95(%)'] - pivot_ult['RAW_mAP@50-95(%)']

    cols_ult = [
        'Class', 'Weather', 'Train_Instances', 'Total_Instances_Per_Weather', 
        'RAW_Precision(%)', 'CLAHE_Precision(%)', 'Diff_Precision',
        'RAW_Recall(%)', 'CLAHE_Recall(%)', 'Diff_Recall',
        'RAW_F1-Score(%)', 'CLAHE_F1-Score(%)', 'Diff_F1',
        'RAW_mAP@50(%)', 'CLAHE_mAP@50(%)', 'Diff_mAP50',
        'RAW_mAP@50-95(%)', 'CLAHE_mAP@50-95(%)', 'Diff_mAP50-95'
    ]
    pivot_ult = pivot_ult[cols_ult].round(2)
    pivot_ult.to_excel("Ultimate_Comparison_Diff.xlsx", index=False)
    print("✅ Created: Ultimate_Comparison_Diff.xlsx")

    df_summary_class = df_summary.copy()
    df_summary_class.rename(columns={'Val_Instances': 'Total_Instances_Per_Weather'}, inplace=True)
    
    pivot_class = df_summary_class.pivot(
        index=['Class', 'Weather', 'Train_Instances', 'Total_Instances_Per_Weather'], 
        columns='Model', 
        values='mAP@50(%)'
    ).reset_index()
    
    pivot_class['Diff(CLAHE - RAW)'] = pivot_class['CLAHE'] - pivot_class['RAW']
    pivot_class.rename(columns={'CLAHE': 'CLAHE_DA', 'RAW': 'RAW_DA'}, inplace=True)
    
    cols_class = ['Class', 'Weather', 'Train_Instances', 'Total_Instances_Per_Weather', 'CLAHE_DA', 'RAW_DA', 'Diff(CLAHE - RAW)']
    pivot_class = pivot_class[cols_class].sort_values(by=['Class', 'Weather']).round(2)
    
    pivot_class['Weight_RAW'] = pivot_class['RAW_DA'] * pivot_class['Total_Instances_Per_Weather']
    pivot_class['Weight_CLAHE'] = pivot_class['CLAHE_DA'] * pivot_class['Total_Instances_Per_Weather']

    weighted_class = pivot_class.groupby('Class').agg({
        'Train_Instances': 'sum',
        'Total_Instances_Per_Weather': 'sum',
        'Weight_RAW': 'sum',
        'Weight_CLAHE': 'sum'
    }).reset_index()

    weighted_class['Weighted_RAW'] = weighted_class['Weight_RAW'] / weighted_class['Total_Instances_Per_Weather']
    weighted_class['Weighted_CLAHE'] = weighted_class['Weight_CLAHE'] / weighted_class['Total_Instances_Per_Weather']
    weighted_class['Weighted_Diff'] = weighted_class['Weighted_CLAHE'] - weighted_class['Weighted_RAW']

    total_val_instances = weighted_class['Total_Instances_Per_Weather'].sum()
    total_train_instances = weighted_class['Train_Instances'].sum()
    total_weight_raw = weighted_class['Weight_RAW'].sum()
    total_weight_clahe = weighted_class['Weight_CLAHE'].sum()

    overall_raw = total_weight_raw / total_val_instances
    overall_clahe = total_weight_clahe / total_val_instances
    overall_diff = overall_clahe - overall_raw

    summary_rows = []
    summary_rows.append({c: "" for c in cols_class})
    summary_rows.append({'Class': '--- TRUNG BÌNH CÓ TRỌNG SỐ THEO CLASS ---', 'Weather': "", 'Train_Instances': "", 'Total_Instances_Per_Weather': "", 'CLAHE_DA': "", 'RAW_DA': "", 'Diff(CLAHE - RAW)': ""})
    
    for _, row in weighted_class.iterrows():
        summary_rows.append({
            'Class': f"Weighted Avg: {row['Class'].upper()}",
            'Weather': 'ALL',
            'Train_Instances': row['Train_Instances'],
            'Total_Instances_Per_Weather': row['Total_Instances_Per_Weather'],
            'CLAHE_DA': round(row['Weighted_CLAHE'], 2),
            'RAW_DA': round(row['Weighted_RAW'], 2),
            'Diff(CLAHE - RAW)': round(row['Weighted_Diff'], 2)
        })

    summary_rows.append({c: "" for c in cols_class})
    summary_rows.append({'Class': '--- TRUNG BÌNH TRỌNG SỐ TỔNG QUAN ---', 'Weather': "", 'Train_Instances': "", 'Total_Instances_Per_Weather': "", 'CLAHE_DA': "", 'RAW_DA': "", 'Diff(CLAHE - RAW)': ""})
    
    summary_rows.append({
        'Class': "OVERALL (Tất cả Xe & Thời tiết)",
        'Weather': 'ALL',
        'Train_Instances': total_train_instances,
        'Total_Instances_Per_Weather': total_val_instances,
        'CLAHE_DA': round(overall_clahe, 2),
        'RAW_DA': round(overall_raw, 2),
        'Diff(CLAHE - RAW)': round(overall_diff, 2)
    })

    summary_df = pd.DataFrame(summary_rows)
    pivot_class_final = pd.concat([pivot_class[cols_class], summary_df], ignore_index=True)
    
    with pd.ExcelWriter("Class_Focused_Aggregated_Comparison.xlsx", engine='openpyxl') as writer:
        pivot_class_final.to_excel(writer, sheet_name='CLAHE_vs_RAW', index=False)
        df_summary_class.to_excel(writer, sheet_name='Detailed_Aggregation', index=False)
        
    print("✅ Created: Class_Focused_Aggregated_Comparison.xlsx")
    print("QUY TRÌNH HOÀN TẤT 100%!")

def main():
    run_evaluation_for_model("RAW_DA")
    run_evaluation_for_model("CLAHE_DA")
    convert_and_aggregate()

if __name__ == '__main__':
    main()