1. Required Directory Structure:
Ensure your workspace is extracted and organized exactly as follows for the pipeline to execute smoothly:
Project_Folder/
│
├── dawn_yolo_dataset/                 # Original Dataset (RAW Images)
├── dawn_clahe_dataset/                # Preprocessed Dataset (CLAHE Images)
├── data_settings/                     # Auto-generated .txt and .yaml config files
├── runs/                              # YOLO output directory (logs and weights)
│
├── SETUP_KFOLD.py                     # Splits dataset for 5-Fold Cross-Validation
├── 2_TRAIN_KFOLD.py                   # Automated mega-training (4 configs x 5 Folds)
├── 2_AUTO_TEST_KFOLD.py               # 5-Fold evaluation and CSV logging
├── AUTO_TEST.py                       # Tests the 8 NMS/DA configuration matrix
├── Detailed_Evaluation_Per_Weather.py # Analyzes errors, calculates weighted mAP, exports to Excel
├── ENSEMBLE_FUSION.py                 # Evaluates Late Fusion & generates 4-in-1 visual grids
└── INFERENCE_Test.py                  # Real-world inference on user Images/Videos
2. End-to-End Pipeline Execution:
Step 0: Environment Setup:
Before executing any scripts, configure your Python environment and install all necessary dependencies to ensure compatibility. Open your terminal in the project root directory and run:

pip install -r requirements.txt

Step 1: Data Splitting (SETUP_KFOLD.py):
Run this script to scan the dawn_yolo_dataset. It uses a fixed seed (42) to shuffle and split the data into 5 equal folds, generating the necessary .yaml configuration files in the data_settings/ directory.

Step 2: Model Training (2_TRAIN_KFOLD.py):
Run this script to automatically map the paths for the CLAHE dataset and initiate the 20 training batches (4 configurations across 5 folds). The best weights (best.pt) are automatically saved to the runs/ directory.

Step 3: Evaluation & In-Depth Analysis:

Run 2_AUTO_TEST_KFOLD.py to extract the average mAP@50 scores across folds and export them to K_Fold_Results_Log.csv.

Run Detailed_Evaluation_Per_Weather.py to isolate visual errors, generate Error Grids, and export the critical Class_Focused_Aggregated_Comparison.xlsx (Weighted mAP score matrix).

Run ENSEMBLE_FUSION.py to execute the Late Fusion algorithm and observe the noise amplification phenomenon via the 4-in-1 output grids.

Step 4: Real-World Inference Demo (INFERENCE_Test.py):
Run this file to test the model on custom images or videos. Modify the TEST_IMAGE or TEST_VIDEO paths inside the script. It will read the input, apply the CLAHE filter (if enabled), perform detection, and output an annotated .mp4 or image file.
