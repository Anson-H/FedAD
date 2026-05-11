#!/usr/bin/env python3
"""
Data processing utilities.
Includes dataset loading, landcover lookup, and per-satellite data assignment.
"""

import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from flcore.config import Config
from flcore.utils import _normalize_probs_dict


class PastTestingDatasetWithGeo(Dataset):
    """Wrap a past-test subset and attach the geo-state snapshot for each sample."""

    def __init__(self, base_subset, geo_states: List):
        if len(base_subset) != len(geo_states):
            raise ValueError(
                f"PastTestingDatasetWithGeo length mismatch: "
                f"base_subset={len(base_subset)}, geo_states={len(geo_states)}"
            )
        self.base_subset = base_subset
        self.geo_states = geo_states

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, idx: int):
        x, y = self.base_subset[idx]
        s = self.geo_states[idx]
        return x, y, s


class CombinedDatasetWithSource(Dataset):
    """Combine training and exemplar samples and label the data source."""

    def __init__(self, base_dataset, train_indices: List[int], exemplar_indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = []
        self.source_labels = []

        for idx in train_indices:
            self.indices.append(idx)
            self.source_labels.append(0)

        for idx in exemplar_indices:
            self.indices.append(idx)
            self.source_labels.append(1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, y = self.base_dataset[original_idx]
        source = self.source_labels[idx]
        return x, y, source


def load_landcover_distribution(file_path, resolution):
    """Load the global landcover distribution table from CSV."""
    n_rows = 180 // resolution
    n_cols = 360 // resolution
    distribution = [
        [{cat: 0.0 for cat in Config.LANDCOVER_CATEGORIES} for _ in range(n_cols)]
        for _ in range(n_rows)
    ]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Landcover distribution file not found: {file_path}")

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["Row"])
            c = int(row["Col"])
            if 0 <= r < n_rows and 0 <= c < n_cols:
                for cat in Config.LANDCOVER_CATEGORIES:
                    distribution[r][c][cat] = float(row[cat])

    return distribution


LANDCOVER_DISTRIBUTION = load_landcover_distribution(
    os.path.join("data", f"global_landcover_distribution_{Config.LANDCOVER_RESOLUTION}.csv"),
    Config.LANDCOVER_RESOLUTION,
)


def get_landcover_distribution(lat_rad: float, lon_rad: float) -> Dict[str, float]:
    """Query landcover probabilities from satellite latitude and longitude."""
    resolution = Config.LANDCOVER_RESOLUTION
    n_rows = 180 // resolution
    n_cols = 360 // resolution

    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    row = int((90 - lat_deg) / resolution)
    row = max(0, min(n_rows - 1, row))

    col = int((lon_deg + 180) / resolution)
    col = col % n_cols

    return _normalize_probs_dict(LANDCOVER_DISTRIBUTION[row][col])


def _load_training_data(data_path, train_ratio, num_classes, image_size=64):
    """Load a training subset and class-wise indices from an ImageFolder dataset."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    all_targets = np.asarray([sample[1] for sample in full_dataset.samples], dtype=int)

    train_indices = []
    rng = np.random.RandomState(Config.DATASET_ASSIGNMENT_SEED_BASE)

    for class_idx in range(num_classes):
        class_samples = np.where(all_targets == class_idx)[0]
        n_train = int(len(class_samples) * train_ratio)
        shuffled = rng.permutation(class_samples)
        train_indices.extend(shuffled[:n_train].tolist())

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_targets = all_targets[train_indices]
    class_indices = [np.where(train_targets == i)[0] for i in range(num_classes)]
    return train_dataset, class_indices


def _load_testing_data(data_path, train_ratio, num_classes, image_size=64):
    """Load a test subset and class-wise indices from an ImageFolder dataset."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    all_targets = np.asarray([sample[1] for sample in full_dataset.samples], dtype=int)

    test_indices = []
    rng = np.random.RandomState(Config.DATASET_ASSIGNMENT_SEED_BASE)

    for class_idx in range(num_classes):
        class_samples = np.where(all_targets == class_idx)[0]
        n_train = int(len(class_samples) * train_ratio)
        shuffled = rng.permutation(class_samples)
        test_indices.extend(shuffled[n_train:].tolist())

    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_targets = all_targets[test_indices]
    class_indices = [np.where(test_targets == i)[0] for i in range(num_classes)]
    return test_dataset, class_indices


def load_eurosat_training_data():
    return _load_training_data(
        Config.EUROSAT_DATA_PATH,
        Config.EUROSAT_TRAIN_RATIO,
        10,
        image_size=64,
    )


def load_nwpu_training_data():
    return _load_training_data(
        Config.NWPU_DATA_PATH,
        Config.NWPU_TRAIN_RATIO,
        45,
        image_size=224,
    )


def load_eurosat_testing_data():
    return _load_testing_data(
        Config.EUROSAT_DATA_PATH,
        Config.EUROSAT_TRAIN_RATIO,
        10,
        image_size=64,
    )


def load_nwpu_testing_data():
    return _load_testing_data(
        Config.NWPU_DATA_PATH,
        Config.NWPU_TRAIN_RATIO,
        45,
        image_size=224,
    )


def load_training_data():
    """Load the training dataset selected by Config.DATASET_NAME."""
    if Config.DATASET_NAME == "EuroSAT":
        return load_eurosat_training_data()
    if Config.DATASET_NAME == "NWPU":
        return load_nwpu_training_data()
    raise ValueError(
        f"Unsupported dataset: {Config.DATASET_NAME}, valid options are 'EuroSAT' or 'NWPU'"
    )


def load_testing_data():
    """Load the testing dataset selected by Config.DATASET_NAME."""
    if Config.DATASET_NAME == "EuroSAT":
        return load_eurosat_testing_data()
    if Config.DATASET_NAME == "NWPU":
        return load_nwpu_testing_data()
    raise ValueError(
        f"Unsupported dataset: {Config.DATASET_NAME}, valid options are 'EuroSAT' or 'NWPU'"
    )


def assign_dataset_to_satellite(
    sat,
    train_data,
    test_data,
    landcover_probs: Dict[str, float] | None = None,
    seed: int | None = None,
    train_batch_size: int = Config.LOCAL_TRAIN_BATCH_SIZE,
    test_batch_size: int = Config.LOCAL_TEST_BATCH_SIZE,
    is_combined: bool = False,
):
    """Assign local train/test subsets to a satellite based on landcover distribution."""
    num_classes = Config.NUM_CLASSES
    landcover_probs = landcover_probs or {cat: 0.0 for cat in Config.LANDCOVER_CATEGORIES}

    probs = np.zeros(num_classes, dtype=float)
    normed = _normalize_probs_dict({
        cat: landcover_probs.get(cat, 0.0) for cat in Config.LANDCOVER_CATEGORIES
    })

    for cat, p in normed.items():
        labels = Config.LANDCOVER_TO_LABELS[cat]
        if labels:
            prob_per_label = p / len(labels)
            for label in labels:
                probs[label] += prob_per_label

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(num_classes) / num_classes

    rng = np.random.default_rng(seed)

    train_dataset, train_class_indices = train_data
    test_dataset, test_class_indices = test_data

    train_class_counts = rng.multinomial(max(1, sat.n_train // 20), probs)
    train_selected_indices: List[int] = []
    for cls_idx, count in enumerate(train_class_counts):
        if count > 0:
            available = len(train_class_indices[cls_idx])
            actual_count = min(count, available)
            if actual_count > 0:
                chosen = rng.choice(train_class_indices[cls_idx], actual_count, replace=False)
                train_selected_indices.extend([int(x) for x in chosen])

    sat.training_data = torch.utils.data.Subset(train_dataset, train_selected_indices)
    sat.training_dataloader = DataLoader(
        sat.training_data,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    if is_combined:
        exemplar_indices = []
        for _, indices in sat.exemplar_indices.items():
            exemplar_indices.extend(indices)

        train_selected_set = set(train_selected_indices)
        exemplar_indices = [idx for idx in exemplar_indices if idx not in train_selected_set]

        sat.exemplar_data = torch.utils.data.Subset(train_dataset, exemplar_indices)
        sat.exemplar_dataloader = (
            DataLoader(sat.exemplar_data, batch_size=train_batch_size, shuffle=True, drop_last=False)
            if exemplar_indices else None
        )

        exemplar_set = set(exemplar_indices)
        if train_selected_set & exemplar_set:
            raise ValueError("Training and exemplar indices overlap")

        sat.training_combined_data = CombinedDatasetWithSource(
            base_dataset=train_dataset,
            train_indices=train_selected_indices,
            exemplar_indices=exemplar_indices,
        )
        sat.training_combined_dataloader = DataLoader(
            sat.training_combined_data,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )

    samples_per_class = min(len(indices) for indices in test_class_indices) // 8
    full_test_class_counts = np.array([samples_per_class] * num_classes, dtype=int)
    full_test_selected_indices: List[int] = []
    for cls_idx, count in enumerate(full_test_class_counts):
        if count > 0:
            available = len(test_class_indices[cls_idx])
            actual_count = min(count, available)
            if actual_count > 0:
                chosen = rng.choice(test_class_indices[cls_idx], actual_count, replace=False)
                full_test_selected_indices.extend([int(x) for x in chosen])

    sat.full_testing_data = torch.utils.data.Subset(test_dataset, full_test_selected_indices)
    sat.full_testing_dataloader = DataLoader(
        sat.full_testing_data,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    current_test_class_counts = rng.multinomial(max(1, sat.n_train // 20), probs)
    sat.current_testing_indices = []
    for cls_idx, count in enumerate(current_test_class_counts):
        if count > 0:
            available = len(test_class_indices[cls_idx])
            actual_count = min(count, available)
            if actual_count > 0:
                chosen = rng.choice(test_class_indices[cls_idx], actual_count, replace=False)
                sat.current_testing_indices.extend([int(x) for x in chosen])

    sat.current_testing_data = torch.utils.data.Subset(test_dataset, sat.current_testing_indices)
    sat.current_testing_dataloader = DataLoader(
        sat.current_testing_data,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    if len(sat.past_testing_indices) > 0:
        past_subset = torch.utils.data.Subset(test_dataset, sat.past_testing_indices)
        if len(sat.past_testing_geo_states) == len(sat.past_testing_indices):
            sat.past_testing_data = PastTestingDatasetWithGeo(
                past_subset,
                sat.past_testing_geo_states,
            )
        else:
            sat.past_testing_data = past_subset
        sat.past_testing_dataloader = DataLoader(
            sat.past_testing_data,
            batch_size=test_batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        sat.past_testing_data = None
        sat.past_testing_dataloader = None

    full_test_dataset = datasets.ImageFolder(root=Config.DATA_PATH, transform=None)
    all_targets = np.asarray([sample[1] for sample in full_test_dataset.samples], dtype=int)

    past_test_class_counts = np.zeros(num_classes, dtype=int)
    for idx in sat.past_testing_indices:
        original_idx = test_dataset.indices[idx]
        past_test_class_counts[all_targets[original_idx]] += 1

    sat.train_dist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(train_class_counts)])
    sat.full_test_dist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(full_test_class_counts)])
    sat.current_test_dist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(current_test_class_counts)])
    sat.past_test_dist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(past_test_class_counts)])
