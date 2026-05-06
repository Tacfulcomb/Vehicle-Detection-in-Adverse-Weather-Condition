import os
import cv2
import torch
import torchvision
import numpy as np
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

print("="*60)
print("🔥 ĐÁNH GIÁ & VISUALIZE MÔ HÌNH HỢP THỂ (ENSEMBLE LATE FUSION)")
print("="*60)

# ==========================================
# CẤU HÌNH THÔNG SỐ (Tùy chỉnh tại đây)
# ==========================================
CONF_THRESH = 0.2
IOU_THRESH = 0.5    
NUM_IMAGES_TO_SAVE = 20  # Số lượng ảnh muốn xuất ra để xem mỗi Fold (tránh tràn ổ cứng)
VIS_DIR = "runs/ensemble_vis" # Thư mục lưu ảnh

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

# ==========================================
# HÀM BỔ TRỢ
# ==========================================
def apply_clahe_to_frame(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def yolo_to_xyxy(yolo_boxes, img_width, img_height):
    boxes = []
    for box in yolo_boxes:
        cls, x_center, y_center, w, h = box
        x1 = (x_center - w / 2) * img_width
        y1 = (y_center - h / 2) * img_height
        x2 = (x_center + w / 2) * img_width
        y2 = (y_center + h / 2) * img_height
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)

# Hàm vẽ Bounding Box chuyên dụng
def draw_boxes(img, boxes, labels, scores=None, title="", color=(0, 255, 0)):
    img_draw = img.copy()
    # Vẽ nền đen mờ cho tiêu đề dễ đọc
    cv2.rectangle(img_draw, (0, 0), (img_draw.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img_draw, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        cls_id = int(labels[i])
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        
        label_text = f"{name}"
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
            
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        # Vẽ nền cho text
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img_draw, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return img_draw

# ==========================================
# THỰC THI CHẤM ĐIỂM & VẼ ẢNH
# ==========================================
if __name__ == '__main__':
    total_map50 = 0.0
    os.makedirs(VIS_DIR, exist_ok=True)

    for fold in range(1, 6):
        print(f"\n🚀 ĐANG CHẤM ĐIỂM FOLD {fold}/5 (Conf: {CONF_THRESH}, IoU: {IOU_THRESH})...")
        fold_vis_dir = os.path.join(VIS_DIR, f"fold_{fold}")
        os.makedirs(fold_vis_dir, exist_ok=True)
        
        model_raw_path = f"runs/detect/KFOLD_RAW_DA_Fold_{fold}/weights/best.pt"
        model_clahe_path = f"runs/detect/KFOLD_CLAHE_DA_Fold_{fold}/weights/best.pt"
        val_txt_path = f"data_settings/fold_{fold}_val.txt"

        if not os.path.exists(model_raw_path) or not os.path.exists(model_clahe_path):
            print(f"❌ Thiếu file best.pt ở Fold {fold}, bỏ qua!")
            continue

        model_raw = YOLO(model_raw_path)
        model_clahe = YOLO(model_clahe_path)
        metric = MeanAveragePrecision(iou_type="bbox")

        with open(val_txt_path, 'r') as f:
            val_images = f.read().strip().split('\n')

        preds = []
        targets = []
        saved_count = 0

        for img_path in tqdm(val_images, desc=f"Xử lý ảnh Fold {fold}"):
            img_raw = cv2.imread(img_path)
            if img_raw is None: continue
            
            h, w, _ = img_raw.shape
            img_clahe = apply_clahe_to_frame(img_raw)

            # --- A. LẤY DỰ ĐOÁN ---
            res_raw = model_raw.predict(img_raw, conf=0.25, iou=0.5, verbose=False)[0]
            res_clahe = model_clahe.predict(img_clahe, conf=0.25, iou=0.5, verbose=False)[0]

            boxes_raw = res_raw.boxes.data.cpu()
            boxes_clahe = res_clahe.boxes.data.cpu()

            # --- B. HỢP THỂ (ENSEMBLE FUSION) ---
            all_boxes = torch.cat([boxes_raw, boxes_clahe], dim=0)
            mask_conf = all_boxes[:, 4] > CONF_THRESH
            all_boxes = all_boxes[mask_conf]

            if len(all_boxes) > 0:
                max_coord = 10000.0  
                offsets = all_boxes[:, 5] * max_coord
                boxes_for_nms = all_boxes[:, :4] + offsets[:, None] 
                
                keep_idx = torchvision.ops.nms(boxes_for_nms, all_boxes[:, 4], iou_threshold=IOU_THRESH)
                kept_boxes = all_boxes[keep_idx]
                
                final_boxes = kept_boxes[:, :4]
                final_scores = kept_boxes[:, 4]
                final_labels = kept_boxes[:, 5].to(torch.int64)
            else:
                final_boxes = torch.empty((0, 4))
                final_scores = torch.empty((0,))
                final_labels = torch.empty((0,), dtype=torch.int64)

            preds.append(dict(boxes=final_boxes, scores=final_scores, labels=final_labels))

            # --- C. ĐỌC NHÃN GỐC (GROUND TRUTH) ---
            label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            gt_boxes_yolo, gt_labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_labels.append(int(parts[0]))
                            gt_boxes_yolo.append([float(p) for p in parts])
            
            if len(gt_boxes_yolo) > 0:
                gt_boxes = yolo_to_xyxy(gt_boxes_yolo, w, h)
                targets.append(dict(boxes=gt_boxes, labels=torch.tensor(gt_labels, dtype=torch.int64)))
            else:
                gt_boxes = torch.empty((0, 4))
                targets.append(dict(boxes=gt_boxes, labels=torch.empty((0,), dtype=torch.int64)))

            # ==========================================
            # D. VISUALIZE 4-IN-1 (LƯU ẢNH SO SÁNH)
            # ==========================================
            if saved_count < NUM_IMAGES_TO_SAVE:
                # Lọc lại box Raw & Clahe theo Threshold để vẽ cho đỡ rác
                draw_raw_boxes = boxes_raw[boxes_raw[:, 4] > CONF_THRESH]
                draw_clahe_boxes = boxes_clahe[boxes_clahe[:, 4] > CONF_THRESH]

                # 1. Vẽ 4 khung hình
                img_gt = draw_boxes(img_raw, gt_boxes, gt_labels, scores=None, title="1. GROUND TRUTH (Nhan Goc)", color=(0, 255, 0)) # Xanh lá
                img_ens = draw_boxes(img_raw, final_boxes, final_labels, final_scores, title="2. ENSEMBLE FINAL", color=(255, 0, 255)) # Tím
                img_r = draw_boxes(img_raw, draw_raw_boxes[:, :4], draw_raw_boxes[:, 5], draw_raw_boxes[:, 4], title="3. RAW DETECT", color=(0, 165, 255)) # Cam
                img_c = draw_boxes(img_clahe, draw_clahe_boxes[:, :4], draw_clahe_boxes[:, 5], draw_clahe_boxes[:, 4], title="4. CLAHE DETECT", color=(255, 255, 0)) # Xanh lơ

                # 2. Ghép thành Grid 2x2
                top_row = np.hstack((img_gt, img_ens))
                bottom_row = np.hstack((img_r, img_c))
                grid_img = np.vstack((top_row, bottom_row))

                # 3. Resize lại cho đỡ nặng (nếu ảnh gốc to quá)
                grid_img = cv2.resize(grid_img, (grid_img.shape[1] // 2, grid_img.shape[0] // 2))

                # 4. Lưu ảnh
                img_name = os.path.basename(img_path)
                save_path = os.path.join(fold_vis_dir, f"vis_{img_name}")
                cv2.imwrite(save_path, grid_img)
                saved_count += 1

        # Nạp và tính điểm mAP
        metric.update(preds, targets)
        result = metric.compute()
        map_50 = result['map_50'].item() * 100
        print(f"🏆 mAP@50 của Ensemble tại Fold {fold}: {map_50:.2f}%")
        
        total_map50 += map_50
        metric.reset()

    print("\n" + "="*60)
    print(f"🌟 ĐIỂM ENSEMBLE K-FOLD TRUNG BÌNH: {total_map50 / 5.0:.2f}% (Conf: {CONF_THRESH}, IoU: {IOU_THRESH}) 🌟")
    print(f"📸 Ảnh so sánh đã được lưu tại thư mục: {VIS_DIR}")
    print("="*60)