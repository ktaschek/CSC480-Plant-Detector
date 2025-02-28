#!/usr/bin/env python3
"""
train.py

This script trains a deep neural network classifier for plant species recognition.
Key aspects inspired by the paper:
 • Uses a deep CNN (here, ResNet-50 fine-tuned from ImageNet) as feature extractor/classifier.
 • Applies data augmentation mimicking RandomResizedCrop, horizontal/vertical flip, and brightness/contrast jitter.
 • Uses SGD with momentum and a ReduceLROnPlateau learning rate scheduler.
 • Implements gradient accumulation to achieve an effective batch size of 128 (e.g., accumulate 4 mini-batches of size 32).
 
Usage example:
    python train.py --data_dir /path/to/prepared_dataset --epochs 100 --batch_size 32 --lr 0.001 --output_dir ./checkpoints
"""

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def parse_args():
    parser = argparse.ArgumentParser(description="Train PlantNet DNN classifier.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to prepared dataset with train/val/test folders")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--arch", type=str, default="resnet50",
                        help="Model architecture (default: resnet50)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (e.g., 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of mini-batches to accumulate (default: 4 for effective batch size of 128)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), test_transforms)
    test_dataset  = datasets.ImageFolder(os.path.join(args.data_dir, "test"), test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Load pre-trained model (ResNet-50 as baseline)
    if args.arch.lower() == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print("Unknown architecture, defaulting to ResNet-50.")
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=2, verbose=True)

    best_val_loss = np.inf
    num_steps = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad()
        epoch_start = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) / args.accumulation_steps
            loss.backward()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * args.accumulation_steps * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_steps += 1
            
            if num_steps % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Time: {time.time()-epoch_start:.2f}s")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Validation - Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")
        
        # Step scheduler based on validation loss
        scheduler.step(val_epoch_loss)
        
        # Save checkpoint if validation loss improved
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_epoch_loss,
                'val_acc': val_epoch_acc.item()
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Evaluate on test set
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_acc = test_corrects.double() / len(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
