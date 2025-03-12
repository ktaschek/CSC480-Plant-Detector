"""
Usage:
    python prepare.py --input_dir /path/to/raw_dataset --output_dir /path/to/processed_dataset --target_size 224
"""

import os
import argparse
import cv2
import json

def process_images_for_split(input_split_dir, output_split_dir, target_size):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    # Process each class folder in this split
    for class_name in sorted(os.listdir(input_split_dir)):
        class_input_path = os.path.join(input_split_dir, class_name)
        if not os.path.isdir(class_input_path):
            continue
        class_output_path = os.path.join(output_split_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        image_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(valid_ext)]
        for fname in image_files:
            src_path = os.path.join(class_input_path, fname)
            img = cv2.imread(src_path)
            if img is None:
                print(f"Warning: Could not read {src_path}")
                continue
            # Resize image to target_size x target_size
            img_resized = cv2.resize(img, (target_size, target_size))
            dest_path = os.path.join(class_output_path, fname)
            cv2.imwrite(dest_path, img_resized)

def main():
    parser = argparse.ArgumentParser(description="Process and resize PlantNet images.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing 'train', 'val', and 'test' folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory where processed images will be saved.")
    parser.add_argument("--target_size", type=int, default=224,
                        help="Target image size (square), default is 224.")
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for split in splits:
        input_split = os.path.join(args.input_dir, split)
        output_split = os.path.join(args.output_dir, split)
        os.makedirs(output_split, exist_ok=True)
        print(f"Processing {split} split...")
        process_images_for_split(input_split, output_split, args.target_size)

    # Build class mapping based on the 'train' folder (assuming all classes are present there)
    train_dir = os.path.join(args.input_dir, 'train')
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_names = sorted(class_names)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    mapping_path = os.path.join(args.output_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"Class mapping saved to {mapping_path}")
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
