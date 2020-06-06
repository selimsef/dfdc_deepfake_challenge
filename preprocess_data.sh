DATA_ROOT=$1
echo "Extracting bounding boxes from original videos"
PYTHONPATH=. python preprocessing/detect_original_faces.py --root-dir $DATA_ROOT

echo "Extracting crops as pngs"
PYTHONPATH=. python preprocessing/extract_crops.py --root-dir $DATA_ROOT --crops-dir crops

echo "Extracting landmarks"
PYTHONPATH=. python preprocessing/generate_landmarks.py --root-dir $DATA_ROOT

echo "Extracting SSIM masks"
PYTHONPATH=. python preprocessing/generate_diffs.py --root-dir $DATA_ROOT

echo "Generate folds"
PYTHONPATH=. python preprocessing/generate_folds.py --root-dir $DATA_ROOT --out folds.csv