import cv2
import os
from ultralytics import YOLO

# =========================================================
# HÀM BỔ TRỢ: TỰ ĐỘNG LỌC SƯƠNG BẰNG CLAHE CHO TỪNG FRAME
# =========================================================
def apply_clahe_to_frame(img):
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

# =========================================================
# HÀM 1: TEST TRÊN ẢNH ĐƠN (PICTURE)
# =========================================================
def detect_vehicle_picture(image_path, model_path, use_clahe=False, use_nms=True):
    print(f"\n ĐANG TEST ẢNH: {image_path}")
    print(f" Model: {model_path} | CLAHE: {use_clahe} | NMS: {use_nms}")
    
    # 1. Load model
    model = YOLO(model_path)
    
    # 2. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(" Lỗi: Không đọc được ảnh. Vui lòng check lại đường dẫn!")
        return

    # 3. Tiền xử lý (Nếu dùng model CLAHE thì phải bật cái này)
    if use_clahe:
        img = apply_clahe_to_frame(img)

    # 4. Setup NMS
    iou_thresh = 0.4 if use_nms else 1.0
    conf_thresh = 0.4 if use_nms else 0.001

    # 5. Cho AI dự đoán
    results = model.predict(source=img, iou=iou_thresh, conf=conf_thresh, save=True, save_dir="runs/detect/predict_images")
    
    print("✅ Xong! Kết quả ảnh đã được lưu vào thư mục: runs/detect/train.../predict")

# =========================================================
# HÀM 2: TEST TRÊN VIDEO
# =========================================================
def detect_vehicle_video(video_path, model_path, use_clahe=False, use_nms=True, output_name="output_video.mp4"):
    print(f"\n ĐANG TEST VIDEO: {video_path}")
    print(f" Model: {model_path} | CLAHE: {use_clahe} | NMS: {use_nms}")

    # 1. Load model
    model = YOLO(model_path)

    # 2. Mở Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Lỗi: Không mở được video. Check lại đường dẫn!")
        return

    # 3. Lấy thông số video để xuất file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    frame_count = 0
    print("⏳ Đang xử lý từng frame... (Bấm 'q' ở cửa sổ video để dừng sớm)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Tiền xử lý khung hình nếu xài model CLAHE
        if use_clahe:
            frame = apply_clahe_to_frame(frame)

        # Setup NMS
        iou_thresh = 0.6 if use_nms else 1.0
        conf_thresh = 0.5 if use_nms else 0.001

        # Predict trên khung hình hiện tại
        results = model.predict(source=frame, iou=iou_thresh, conf=conf_thresh, verbose=False)
        
        # Vẽ khung (Bounding Box) lên frame
        annotated_frame = results[0].plot()

        # Ghi frame vào file output
        out.write(annotated_frame)
        
        # Mở cửa sổ xem trực tiếp 
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Đã xử lý {frame_count} frames...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Xong! Video kết quả đã được lưu tại: {output_name}")

# =========================================================
# KHU VỰC CHẠY THỬ (MAIN)
# =========================================================
if __name__ == '__main__':
    WEIGHT_PATH= "runs/detect/RAW_DA_True/weights/best.pt" 
    
    TEST_IMAGE = "sand_storm_test.jpg" 
    TEST_VIDEO = "Foggy.mp4" 


    # ----- THỰC THI -----
    detect_vehicle_picture(image_path=TEST_IMAGE,model_path=WEIGHT_PATH,use_clahe=False,use_nms=True);
    detect_vehicle_video(video_path=TEST_VIDEO,model_path=WEIGHT_PATH,use_clahe=False,use_nms=True,output_name="foggy_processed_NMS_DA_RAW.mp4")