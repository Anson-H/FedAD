#!/usr/bin/env python3
"""
Evaluation and recording module
Used for federated learning model evaluation and result recording.
"""

import os
import csv
from typing import List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

import flcore.satellite as Satellite
from flcore.config import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def local_evaluation(
    sat: "Satellite",
    device: torch.device = DEVICE,
    mode: str = None,
) -> Tuple[float, float, int]:
    """Evaluate one satellite model and return loss, correct predictions, and sample count."""

    sat.local_model = sat.local_model.to(device)
    sat.local_model.eval()

    supports_geo = hasattr(sat.local_model, "set_geo_state")
    if supports_geo and mode != "past_test":
        lc_vec = [sat.collected_landcover.get(cat, 0.0) for cat in Config.LANDCOVER_CATEGORIES]
        sat.local_model.set_geo_state(lc_vec)

    if mode == "train":
        dataloader = sat.training_dataloader
    elif mode == "full_test":
        dataloader = sat.full_testing_dataloader
    elif mode == "current_test":
        dataloader = sat.current_testing_dataloader
    elif mode == "past_test":
        dataloader = sat.past_testing_dataloader
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    
    total_loss = 0.0
    total_correct = 0
    total_size = 0

    if dataloader is None or len(dataloader.dataset) == 0:
        raise ValueError(f"DataLoader is empty, evaluation cannot proceed (mode={mode})")

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, s_batch = batch
                x, y = x.to(device), y.to(device)
                if supports_geo:
                    logits = sat.local_model(x, geo_state=s_batch.to(device))
                else:
                    logits = sat.local_model(x)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = sat.local_model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == y).sum().item())
            total_loss += float(loss.item()) * x.shape[0]
            total_size += x.shape[0]

    sat.local_model = sat.local_model.to('cpu')

    return total_loss, total_correct, total_size


def global_evaluation_and_recording(
    eligible_sats: List["Satellite"],
    mode: str,
    loss_history: List[float],
    accuracy_history: List[float],
    desc: str,
    device: torch.device = DEVICE
) -> Tuple[float, float]:
    """Evaluate all eligible satellites, record the result, and return global loss/accuracy."""
    total_loss = 0
    total_correct = 0
    total_size = 0
    
    for sat in tqdm(eligible_sats, desc=desc, unit="sat"):
        loss, correct, size = local_evaluation(sat, device=device, mode=mode)
        total_loss += loss
        total_correct += correct
        total_size += size
    
    if total_size > 0:
        avg_loss = total_loss / total_size
        avg_accuracy = total_correct / total_size
        loss_history.append(avg_loss)
        accuracy_history.append(avg_accuracy)
        return avg_loss, avg_accuracy
    else:
        raise ValueError(f"Total sample count must be greater than 0, current value is {total_size}")
    
    return 0.0, 0.0


def save_results_to_csv(
    full_testing_accuracy_history: List[float],
    past_testing_accuracy_history: List[float],
    temporal_knowledge_retention_history: List[float],
    current_testing_accuracy_history: List[float],
    algorithm_name: str,
    dataset_name: str,
    results_dir: str = 'results'
):
    """Save evaluation histories to CSV files."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    
    if full_testing_accuracy_history:
        csv_filename = os.path.join(results_dir, f'full_testing_accuracy_{algorithm_name}_{dataset_name}_{timestamp}.csv')
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Global_Testing_Accuracy'])
            for round_idx, accuracy in enumerate(full_testing_accuracy_history, start=1):
                writer.writerow([round_idx, accuracy])
        
        print("\n" + "="*50 + "Save global test accuracy history" + "="*50)
        print(f"Saved to: {csv_filename}")
        print(f"Total communication rounds: {len(full_testing_accuracy_history)}")
        if full_testing_accuracy_history:
            print(f"Initial accuracy: {full_testing_accuracy_history[0]:.4f}")
            print(f"Final accuracy: {full_testing_accuracy_history[-1]:.4f}")
            print(f"Best accuracy: {max(full_testing_accuracy_history):.4f}")
    else:
        print("\nWarning: no global test accuracy history available")

    if past_testing_accuracy_history:
        csv_filename = os.path.join(results_dir, f'past_testing_accuracy_{algorithm_name}_{dataset_name}_{timestamp}.csv')
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Past_Testing_Accuracy'])
            for round_idx, accuracy in enumerate(past_testing_accuracy_history, start=1):
                writer.writerow([round_idx, accuracy])
        
        print("\n" + "="*50 + "Save past test accuracy history" + "="*50)
        print(f"Saved to: {csv_filename}")
        print(f"Total communication rounds: {len(past_testing_accuracy_history)}")
        if past_testing_accuracy_history:
            print(f"Initial accuracy: {past_testing_accuracy_history[0]:.4f}")
            print(f"Final accuracy: {past_testing_accuracy_history[-1]:.4f}")
            print(f"Best accuracy: {max(past_testing_accuracy_history):.4f}")
    else:
        print("\nWarning: no past test accuracy history available")

    if current_testing_accuracy_history:
        csv_filename = os.path.join(results_dir, f'local_testing_accuracy_{algorithm_name}_{dataset_name}_{timestamp}.csv')
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Local_Testing_Accuracy'])
            for round_idx, accuracy in enumerate(current_testing_accuracy_history, start=1):
                writer.writerow([round_idx, accuracy])
        
        print("\n" + "="*50 + "Save local test accuracy history" + "="*50)
        print(f"Saved to: {csv_filename}")
        print(f"Total communication rounds: {len(current_testing_accuracy_history)}")
        if current_testing_accuracy_history:
            print(f"Initial accuracy: {current_testing_accuracy_history[0]:.4f}")
            print(f"Final accuracy: {current_testing_accuracy_history[-1]:.4f}")
            print(f"Best accuracy: {max(current_testing_accuracy_history):.4f}")
    else:
        print("\nWarning: no local test accuracy history available")

    if temporal_knowledge_retention_history:
        csv_filename = os.path.join(results_dir, f'temporal_knowledge_retention_{algorithm_name}_{dataset_name}_{timestamp}.csv')
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Temporal_Knowledge_Retention'])
            for round_idx, retention in enumerate(temporal_knowledge_retention_history, start=1):
                writer.writerow([round_idx, retention])
        
        print("\n" + "="*50 + "Temporal Knowledge Retention History" + "="*50)
        print(f"Saved to: {csv_filename}")
        print(f"Total records: {len(temporal_knowledge_retention_history)}")
        if temporal_knowledge_retention_history:
            print(f"Initial retention: {temporal_knowledge_retention_history[0]:.4f}")
            print(f"Final retention: {temporal_knowledge_retention_history[-1]:.4f}")
            print(f"Best retention: {max(temporal_knowledge_retention_history):.4f}")
            print(f"Lowest retention: {min(temporal_knowledge_retention_history):.4f}")
    else:
        print("\nWarning: no temporal knowledge retention history available")
