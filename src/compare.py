"""
Usage:
    python compare.py --checkpoint_dir ./checkpoints --val_dir ./prepared_dataset/val --num_classes 100 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

def load_model_from_checkpoint(checkpoint_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    return model

def get_model_details(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_modules = sum(1 for _ in model.modules())
    return total_params, trainable_params, num_modules

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    avg_loss = running_loss / total_samples
    accuracy = running_corrects.double() / total_samples
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints by displaying parameter details and validation performance.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint (.pth or .tar) files")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Validation dataset directory (ImageFolder structure)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in your dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()
    
    # Define validation transforms (adjust normalization if needed)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])
    
    # Create the validation dataset and DataLoader
    val_dataset = datasets.ImageFolder(args.val_dir, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir)
                        if f.endswith(".pth") or f.endswith(".tar")]
    if not checkpoint_files:
        print("No checkpoint files found in", args.checkpoint_dir)
        return
    
    summary = []
    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt)
        print("========================================")
        print(f"Checkpoint: {ckpt_path}")
        try:
            model = load_model_from_checkpoint(ckpt_path, args.num_classes)
        except Exception as e:
            print(f"Error loading checkpoint {ckpt}: {e}")
            continue
        
        model = model.to(device)
        total_params, trainable_params, num_modules = get_model_details(model)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print("Total parameters: {:,}".format(total_params))
        print("Trainable parameters: {:,}".format(trainable_params))
        print("Number of modules (layers): {}".format(num_modules))
        print("Validation Loss: {:.4f}".format(val_loss))
        print("Validation Accuracy: {:.4f}".format(val_acc))
        
        summary.append((ckpt, total_params, trainable_params, num_modules, val_loss, val_acc.item()))
        print("========================================\n")
    
    # Print a summary table
    print("\nSummary:")
    header = "{:<30} {:>15} {:>20} {:>20} {:>15} {:>15}".format("Checkpoint", "Total Params", "Trainable Params", "Num Modules", "Val Loss", "Val Acc")
    print(header)
    print("-" * len(header))
    for (ckpt, total, trainable, modules, loss, acc) in summary:
        row = "{:<30} {:>15,} {:>20,} {:>20} {:>15.4f} {:>15.4f}".format(ckpt, total, trainable, modules, loss, acc)
        print(row)

if __name__ == "__main__":
    main()
