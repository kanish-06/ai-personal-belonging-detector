import os
import glob
import yaml
import torch
from ultralytics import YOLO

def verify():
    print("=" * 60)
    print("  STEP 1: VERIFY ULTRALYTICS & GPU DETECTION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("[ERROR] CUDA is not available!")
        return False

    print("\n=" * 60)
    print("  STEP 2: CHECK DATASET CONFIGURATION (data.yaml)")
    print("=" * 60)
    data_yaml_path = os.path.join("dataset", "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] {data_yaml_path} not found!")
        return False

    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    expected_names = {0: "keys", 1: "wallet", 2: "phone", 3: "watch", 4: "glasses"}
    cfg_names = data_cfg.get("names", {})
    
    print(f"nc (num classes): {data_cfg.get('nc')}")
    print(f"names: {cfg_names}")

    if data_cfg.get('nc') != 5:
        print("[ERROR] nc in data.yaml is not 5!")
        return False

    if cfg_names != expected_names:
        print("[ERROR] Class names in data.yaml do not match target mapping!")
        return False

    print("data.yaml check PASSED.")

    print("\n=" * 60)
    print("  STEP 3: CHECK TRAIN / VAL IMAGE AND LABEL COUNTS")
    print("=" * 60)

    splits = ['train', 'val']
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    invalid_class_ids = set()

    for split in splits:
        img_dir = os.path.join("dataset", "images", split)
        lbl_dir = os.path.join("dataset", "labels", split)

        images = glob.glob(os.path.join(img_dir, "*.*"))
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        print(f"Split '{split}': {len(images)} images, {len(labels)} label files.")

        if len(images) == 0:
            print(f"[ERROR] No images found in {img_dir}!")
            return False

        # Verify label contents
        for lbl_path in labels:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    else:
                        invalid_class_ids.add(cls_id)

    print("\nLabel Distribution Across Dataset:")
    for cls_id, name in expected_names.items():
        print(f"  Class {cls_id} ({name}): {class_counts[cls_id]} bounding boxes")

    if invalid_class_ids:
        print(f"[ERROR] Found invalid class IDs in labels: {invalid_class_ids}")
        return False

    print("\nALL PRE-TRAINING CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify()
    if not success:
        exit(1)
