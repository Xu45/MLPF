#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train_mlp_12axis_4regions.py

MLP training script for 12-axis signals with 5-dimensional regression output.
The script trains one MLP model per region and saves models, scalers, and figures.

Output dimensions:
    [Fx, Fy, Fz, x, y]
"""

import os
import random
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from get_data import *


# =============================================================================
# Model Definition
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for regression."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, output_dim: int = 5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_model(model: nn.Module, loader: DataLoader):
    """Evaluate model using MSE and MAE."""
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_preds.append(output)
            all_targets.append(target)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = criterion(all_preds, all_targets).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()

    accuracy = np.sum(
        all_preds.cpu().numpy() == all_targets.cpu().numpy()
    ) / len(all_preds) * 100

    print(f"Model Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae, accuracy


def get_predictions(model: nn.Module, loader: DataLoader):
    """Collect predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_preds.append(output)
            all_targets.append(target)

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def calculate_metrics(predictions, targets):
    """Compute regression metrics."""
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, mae, r2


# =============================================================================
# Training Routine
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
):
    """Train MLP model."""
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    mse_hist, mae_hist, r2_hist = [], [], []

    for epoch in range(num_epochs):
        batch_losses = []
        batch_mse, batch_mae, batch_r2 = [], [], []

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            tgt = target.detach().numpy()
            out = output.detach().numpy()
            batch_mse.append(mean_squared_error(tgt, out))
            batch_mae.append(mean_absolute_error(tgt, out))
            batch_r2.append(r2_score(tgt, out))

        train_losses.append(np.mean(batch_losses))
        mse_hist.append(np.mean(batch_mse))
        mae_hist.append(np.mean(batch_mae))
        r2_hist.append(np.mean(batch_r2))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.6f}")

    return train_losses, mse_hist, mae_hist, r2_hist


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":

    for region in range(1, 5):
        print(f"Train: Region {region}")

        task = "5"   # &&&
        base_dir = os.path.join(
            "train_12axis_4regions", task, "init_train", "MLP"
        )
        os.makedirs(base_dir, exist_ok=True)

        root_dir = "truedata/real/dataset_regions"

        # ---------------------------------------------------------------------
        # Load CSV Data
        # ---------------------------------------------------------------------

        data_in = pd.read_csv(f"{root_dir}/xhd_in_r{region}.csv")
        data_out = pd.read_csv(f"{root_dir}/xhd_out_r{region}.csv")

        inputs = data_in.values
        targets = data_out.values

        # ---------------------------------------------------------------------
        # Standardization
        # ---------------------------------------------------------------------

        scaler_inputs = StandardScaler()
        scaler_targets = StandardScaler()

        inputs = scaler_inputs.fit_transform(inputs)
        targets = scaler_targets.fit_transform(targets)

        joblib.dump(
            scaler_inputs, f"{base_dir}/scaler_inputs_r{region}.pkl"
        )
        joblib.dump(
            scaler_targets, f"{base_dir}/scaler_targets_r{region}.pkl"
        )

        inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(
            inputs,
            targets,
            test_size=0.25,
            random_state=42,
            shuffle=True,
        )

        train_dataset = SignalDataset_region(inputs_train, targets_train)
        valid_dataset = SignalDataset_region(inputs_valid, targets_valid)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=15, shuffle=True)

        # ---------------------------------------------------------------------
        # Train Model
        # ---------------------------------------------------------------------

        epochs = 200
        mlp_model = MLP(input_dim=12, hidden_dim=128, output_dim=5)

        train_losses, train_mse, train_mae, train_r2 = train_model(
            mlp_model, train_loader, num_epochs=epochs
        )

        torch.save(
            mlp_model.state_dict(),
            f"{base_dir}/mlp_model_r{region}.pth",
        )
        print("Model saved")

        # ---------------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------------

        print("Evaluating MLP Model:")
        evaluate_model(mlp_model, valid_loader)

        train_preds, train_targets = get_predictions(
            mlp_model, train_loader
        )
        test_preds, test_targets = get_predictions(
            mlp_model, valid_loader
        )

        test_mse, test_mae, test_r2 = calculate_metrics(
            test_preds.numpy(), test_targets.numpy()
        )

        print(
            f"Training Set - "
            f"MSE: {np.mean(train_mse):.4f}, "
            f"MAE: {np.mean(train_mae):.4f}, "
            f"Aver R²: {np.mean(train_r2):.4f}"
        )
        print(
            f"Test Set - "
            f"MSE: {test_mse:.4f}, "
            f"MAE: {test_mae:.4f}, "
            f"R²: {test_r2:.4f}"
        )

        # ---------------------------------------------------------------------
        # Inverse Transform
        # ---------------------------------------------------------------------

        train_preds = scaler_targets.inverse_transform(train_preds.numpy())
        train_targets = scaler_targets.inverse_transform(train_targets.numpy())
        test_preds = scaler_targets.inverse_transform(test_preds.numpy())
        test_targets = scaler_targets.inverse_transform(test_targets.numpy())

        # ---------------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------------

        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(10, 10)

        ax0 = fig.add_subplot(gs[0:5, 0:6])
        ax0.plot(train_losses, label="Training Loss")
        ax0.set_title("Training Loss Curve")
        ax0.grid()
        ax0.legend()

        ax1 = fig.add_subplot(gs[0:3, 6:])
        ax1.plot(train_mse, label="MSE")
        ax1.plot(train_mae, label="MAE")
        ax1.plot(train_r2, label="R²")
        ax1.set_title("Training Metrics")
        ax1.grid()
        ax1.legend()

        test_table = pd.DataFrame(
            {"Test MSE": [test_mse], "Test MAE": [test_mae], "Test R²": [test_r2]}
        )

        ax2 = fig.add_subplot(gs[3:5, 6:])
        ax2.axis("off")
        ax2.table(
            cellText=test_table.values,
            colLabels=test_table.columns,
            loc="center",
        )

        titles = ["Fx", "Fy", "Fz", "x", "y"]
        for i in range(5):
            ax = fig.add_subplot(gs[5:, 2 * i : 2 * i + 2])
            ax.scatter(test_targets[:, i], test_preds[:, i])

            if i <= 2:
                ax.set_xlim(0, 6)
                ax.set_ylim(0, 6)
            else:
                ax.set_xlim(-30, 30)
                ax.set_ylim(-30, 30)

            x = np.linspace(*ax.get_xlim(), 100)
            ax.plot(x, x, "r--")
            ax.set_title(titles[i])
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.grid()

        plt.suptitle(f"MLP property of Region {region}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/mlp_result_r{region}.png", bbox_inches="tight")
        plt.close()
