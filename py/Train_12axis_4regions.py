#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train_12axis_4regions.py

Training script for 12-axis signal classification using a ResNet-based CNN.

Features:
- 12-channel input signals
- 4-region classification
- ResNet backbone
- CrossEntropy loss
- Confusion matrix visualization
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from get_data import SignalDataset
from ResNet import *
from labml_nn.resnet import ResNetBase


# =============================================================================
# Utilities
# =============================================================================

def draw_loss_curve(
    train_loss: list,
    val_loss: list,
    save_path: str,
    name: str,
):
    """Save loss values and plot training / validation loss curve."""
    os.makedirs(save_path, exist_ok=True)

    txt_path = os.path.join(save_path, f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write("train_loss\n")
        for loss in train_loss:
            f.write(f"{loss}\n")
        f.write("val_loss\n")
        for loss in val_loss:
            f.write(f"{loss}\n")

    plt.figure()
    plt.title(name, fontsize=15, fontweight="bold")
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="red")
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_path, f"{name}.png"))
    plt.close()


# =============================================================================
# Model Definition
# =============================================================================

def ResNetModel(
    n_blocks: List[int],
    n_channels: List[int],
    bottlenecks_channels: Optional[List[int]] = None,
    in_channels: int = 3,
    first_kernel_size: int = 3,
):
    """
    Create a ResNet-based classification model.
    """
    base = ResNetBase(
        n_blocks,
        n_channels,
        bottlenecks_channels,
        in_channels=in_channels,
        first_conv_kernel_size=first_kernel_size,
    )

    classification = nn.Linear(n_channels[-1], 4)   # &&&
    model = nn.Sequential(base, classification)

    return model


# =============================================================================
# Training Script
# =============================================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = "5"   # &&&
    base_dir = os.path.join("train_12axis_4regions", task, "init_train", "CNN")
    os.makedirs(base_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    data_CNN = np.load("truedata/real/dataset_regions/train_dataset_CNN.npy")
    signals = data_CNN[:, :12]
    labels = data_CNN[:, 12]

    signals_train, signals_valid, labels_train, labels_valid = train_test_split(
        signals,
        labels,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    train_dataset = SignalDataset(signals_train, labels_train)
    valid_dataset = SignalDataset(signals_valid, labels_valid)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

    # -------------------------------------------------------------------------
    # Model / Loss / Optimizer
    # -------------------------------------------------------------------------

    model = ResNetModel(
        n_blocks=[4, 4, 4, 4],
        n_channels=[32, 64, 128, 256],
        in_channels=12,
        first_kernel_size=6,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    epochs = 70
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, gt in train_loader:
            if torch.isnan(inputs).any():
                print("NaN values found in input signals")

            inputs = inputs.unsqueeze(1).permute(0, 2, 1).to(device)
            gt = gt.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        print(f"epoch: {epoch}, loss: {avg_train_loss}")

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, gt in valid_loader:
                inputs = inputs.unsqueeze(1).permute(0, 2, 1).to(device)
                gt = gt.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += gt.size(0)
                correct += (predicted == gt).sum().item()

                if epoch == epochs - 1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(gt.cpu().numpy())

            val_loss.append(criterion(outputs, gt).item())

        print(f"epoch: {epoch}, Valid Accuracy: {100 * correct / total:.2f}%")

        # ---------------------------------------------------------------------
        # Confusion Matrix (last epoch only)
        # ---------------------------------------------------------------------

        if epoch == epochs - 1:
            conf_mat = confusion_matrix(all_labels, all_preds)
            row_sums = conf_mat.sum(axis=1)
            normalized_conf_mat = conf_mat / row_sums[:, None] * 100

            label_names = list(range(4))   # &&&

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                normalized_conf_mat,
                annot=True,
                cmap="Blues",
                fmt=".2f",
                xticklabels=label_names,
                yticklabels=label_names,
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix 12-axis / 4-regions")
            plt.savefig(os.path.join(base_dir, "Confusion_matrix_deg.png"))
            plt.close()

            draw_loss_curve(
                train_loss,
                val_loss,
                base_dir,
                "train_val_loss_deg",
            )

    # -------------------------------------------------------------------------
    # Save Model
    # -------------------------------------------------------------------------

    torch.save(
        model.state_dict(),
        os.path.join(base_dir, "model_deg.pth"),
    )
