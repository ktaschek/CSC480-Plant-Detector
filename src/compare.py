#!/usr/bin/env python3
"""
model_details.py

This script iterates over all checkpoint files in a specified directory and prints parameter details for each model,
such as:
  - Total number of parameters
  - Trainable parameters
  - Total number of modules (layers)

It handles checkpoints saved as a dictionary (with keys such as "model_state_dict") as well as
checkpoints saved directly as a state_dict.

Usage:
    python compare.py --checkpoint_dir ./checkpoints --num_classes 100
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

def load_model_from_checkpoint(checkpoint_path, num_classes):
    """
    Loads a ResNet-50 model with the final FC layer adjusted to num_classes.
    This function handles checkpoints saved as a dictionary (with a "model_state_dict" key)
    or as a state_dict directly.
    """
    # Create a ResNet-50 model instance (adjust if you use a different architecture)
    model = models.resnet50(pretrained=False)
    # Replace the final fully connected layer for your number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    return model

def print_model_details(model):
    """
    Prints details about the model including:
      - Total number of parameters
      - Trainable parameters
      - Total number of modules (layers)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_modules = sum(1 for _ in model.modules())
    
    print("Total parameters: {:,}".format(total_params))
    print("Trainable parameters: {:,}".format(trainable_params))
    print("Number of modules (layers): {}".format(num_modules))
    # Uncomment below to print the keys in the state_dict:
    # print("State dict keys:")
    # for key in model.state_dict().keys():
    #     print("  ", key)

def main():
    parser = argparse.ArgumentParser(description="Display model parameter details for each checkpoint in a folder.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint (.pth or .tar) files")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in your dataset")
    args = parser.parse_args()
    
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir)
                        if f.endswith(".pth") or f.endswith(".tar")]
    
    if not checkpoint_files:
        print("No checkpoint files found in", args.checkpoint_dir)
        return
    
    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt)
        print("========================================")
        print(f"Checkpoint: {ckpt_path}")
        try:
            model = load_model_from_checkpoint(ckpt_path, args.num_classes)
        except Exception as e:
            print(f"Error loading checkpoint {ckpt}: {e}")
            continue
        
        print_model_details(model)
        print("========================================\n")

if __name__ == "__main__":
    main()
