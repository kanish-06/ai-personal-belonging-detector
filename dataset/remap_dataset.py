import os
import glob
import shutil
import yaml

# The target mapping we need for the pipeline
TARGET_CLASSES = {
    "keys": 0,
    "wallet": 1,
    "phone": 2,
    "watch": 3,
    "glasses": 4
}

# The expected mapping from the Roboflow dataset
# We map the raw dataset names to our target names
NAME_MAPPING = {
    "Car-key": "keys",
    "carkeys": "keys",
    "keys": "keys",
    "Wallet": "wallet",
    "wallet": "wallet",
    "mobile-phone": "phone",
    "cellphone": "phone",
    "phone": "phone",
    "watch": "watch",
    "glasses": "glasses",
    "sunglass": "glasses",
    "sunglasses": "glasses"
}

def remap_dataset(raw_dir, output_dir):
    yaml_path = os.path.join(raw_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get the source ID to source name mapping
    source_names = data.get('names', [])
    
    if isinstance(source_names, list):
        source_id_to_name = {i: name for i, name in enumerate(source_names)}
    elif isinstance(source_names, dict):
        source_id_to_name = source_names
    else:
        print("Error: Could not parse names from data.yaml")
        return

    # Create the mapping from source ID to target ID
    source_id_to_target_id = {}
    for src_id, src_name in source_id_to_name.items():
        target_name = NAME_MAPPING.get(src_name)
        if target_name and target_name in TARGET_CLASSES:
            source_id_to_target_id[src_id] = TARGET_CLASSES[target_name]
        else:
            print(f"Warning: Unknown class '{src_name}' will be ignored.")

    print("Class ID Mapping:")
    for src_id, target_id in source_id_to_target_id.items():
        print(f"  {src_id} ({source_id_to_name[src_id]}) -> {target_id} (target)")

    # Process train and val/valid directories
    for split in ['train', 'valid', 'val']:
        raw_images_dir = os.path.join(raw_dir, split, "images")
        raw_labels_dir = os.path.join(raw_dir, split, "labels")
        
        if not os.path.exists(raw_images_dir):
            continue
            
        target_split = 'val' if split in ['valid', 'val'] else 'train'
        out_images_dir = os.path.join(output_dir, "images", target_split)
        out_labels_dir = os.path.join(output_dir, "labels", target_split)
        
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_labels_dir, exist_ok=True)

        # Copy images
        images = glob.glob(os.path.join(raw_images_dir, "*.*"))
        for img_path in images:
            shutil.copy(img_path, out_images_dir)

        # Remap and copy labels
        labels = glob.glob(os.path.join(raw_labels_dir, "*.txt"))
        for label_path in labels:
            out_label_path = os.path.join(out_labels_dir, os.path.basename(label_path))
            with open(label_path, 'r') as f_in, open(out_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    src_id = int(parts[0])
                    if src_id in source_id_to_target_id:
                        target_id = source_id_to_target_id[src_id]
                        parts[0] = str(target_id)
                        f_out.write(" ".join(parts) + "\n")
                        
        print(f"Processed {len(images)} images and {len(labels)} labels for {target_split} split.")

if __name__ == "__main__":
    remap_dataset("dataset/raw_download", "dataset")
    print("\nDataset successfully remapped and moved into the correct folders!")
