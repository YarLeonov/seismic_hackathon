import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloader
from model import UNet3D

def tversky_loss(inputs, targets, alpha=0.7, beta=0.3, smooth=1.0):
    """
    Tversky loss: great for highly imbalanced class detections like thin faults.
    Alpha controls penalty for false positives, Beta for false negatives.
    """
    inputs = torch.sigmoid(inputs)
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    
    true_pos = (inputs_flat * targets_flat).sum()
    false_neg = ((1 - inputs_flat) * targets_flat).sum()
    false_pos = (inputs_flat * (1 - targets_flat)).sum()
    
    tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1 - tversky

def train_baseline():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"D:\DS 2026\seis" # Path to data
    
    print(f"Loading data from {data_dir}...")

    train_loader = get_dataloader(data_dir, batch_size=2) # 2 batches due to 3D size
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 10
    
    print("Initialized architecture. Starting train loop...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            # Loss combining standard BCE for stability and Tversky for imbalance
            bce = nn.BCEWithLogitsLoss()(preds, y)
            tversky = tversky_loss(preds, y, alpha=0.7, beta=0.3)
            loss = bce + tversky
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"unet3d_epoch_{epoch}.pt")
        
    print("Training Complete!")

if __name__ == "__main__":
    train_baseline()
