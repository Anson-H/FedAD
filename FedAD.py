import math
import time
import csv
import os
import glob
import numpy as np
import random
import copy
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm

from flcore.config import Config
from flcore.models import MAC_ResNet18_FedAD, compute_geo_state_tensor
from flcore.satellite import Satellite, initialize_constellation
from flcore.data_utils import (
    load_training_data,
    load_testing_data,
    get_landcover_distribution,
    assign_dataset_to_satellite
)
from flcore.utils import _normalize_probs_dict
from flcore.evaluation_and_recording import (
    local_evaluation,
    global_evaluation_and_recording,
    save_results_to_csv
)

# ============================================================================
# Fix random seeds - to keep experiments reproducible
# ============================================================================
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ============================================================================
# Global experiment configuration
# ============================================================================
Config.ALGORITHM_NAME = "FedAD"  # Base algorithm name; suffixes are appended from ablation and upload settings
Config.set_dataset("NWPU")                       # Can be switched to "EuroSAT" or "NWPU"

FEDAD_DIM = 512                 # Encoder feature dimension
FEDAD_ADAPTER_RANK = 256        # LoRA rank r, aligned with the FedAD adapter capacity
FEDAD_HYPER_HIDDEN = 64         # Hidden width of the geo hypernetwork
# Ablation variants: "full" | "wo_geo" | "wo_ad"
ABLATION_VARIANT = "full"

UPLOAD_MODE = "baseline"

def _get_federated_prefixes(upload_mode: str) -> list:
    """Return parameter-name prefixes that participate in federated aggregation."""
    prefixes = ["shared_encoder."]
    if upload_mode in ("baseline", "upload_adapter"):
        prefixes.append("head.")
    if upload_mode in ("upload_adapter", "no_head_and_upload_adapter"):
        prefixes.extend([
            "geo_adapter.W_down.",
            "geo_adapter.W_up.",
        ])
    return prefixes


FEDERATED_PREFIXES = _get_federated_prefixes(UPLOAD_MODE)

Config.MODEL = MAC_ResNet18_FedAD(
    in_features=3,
    num_classes=Config.NUM_CLASSES,
    dim=FEDAD_DIM,
    adapter_rank=FEDAD_ADAPTER_RANK,
    hyper_hidden=FEDAD_HYPER_HIDDEN,
    landcover_dim=len(Config.LANDCOVER_CATEGORIES),
    ablation_variant=ABLATION_VARIANT,
)
MODEL_ABLATION_CONFIG = Config.MODEL.get_ablation_config()

_ABLATION_SUFFIX = {
    "full": "",
    "wo_geo": "_wo_GEO",
    "wo_ad": "_wo_AD",
}
_UPLOAD_MODE_SUFFIX = {
    "baseline": "",
    "upload_adapter": "_up_A",
    "no_head": "_down_H",
    "no_head_and_upload_adapter": "_up_A_down_H",
}
Config.ALGORITHM_NAME = (
    Config.ALGORITHM_NAME
    + _ABLATION_SUFFIX.get(ABLATION_VARIANT, "")
    + _UPLOAD_MODE_SUFFIX.get(UPLOAD_MODE, "")
)


def get_federated_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract the subset of parameters that participate in federated aggregation."""
    federated_params = {}
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in FEDERATED_PREFIXES):
            federated_params[name] = param
    return federated_params


def select_clients(satellites: List[Satellite], t: float) -> List[Satellite]:
    """Select satellites that have reached their training trigger time."""
    return [sat for sat in satellites if sat.t_training_trigger > 0 and t >= sat.t_training_trigger]


def run_consensus_and_downsampling(satellites: List[Satellite]):
    """Synchronize training duration across satellites and downsample local data accordingly."""
    for i, sat in enumerate(satellites):
        sat.t_train_global = max(sat.t_train, Config.T_MIN_LIMIT)

    for _ in range(Config.S // 2):
        prev_taus = [sat.t_train_global for sat in satellites]
        for i, sat in enumerate(satellites):
            min_val = prev_taus[i]
            for n_idx, _, _, direction in sat.neighbors:
                if direction in ['forward', 'backward']:
                    min_val = min(min_val, prev_taus[n_idx])
            sat.t_train_global = min_val

    for _ in range(Config.P // 2):
        prev_taus = [sat.t_train_global for sat in satellites]
        for i, sat in enumerate(satellites):
            min_val = prev_taus[i]
            for n_idx, _, _, direction in sat.neighbors:
                if direction in ['left', 'right']:
                    min_val = min(min_val, satellites[n_idx].t_train_global)
            sat.t_train_global = min_val

    t_setup = Config.ALPHA_WORKER * Config.T_WORKER
    for sat in satellites:
        denominator = max(Config.T_MIN_LIMIT, sat.t_train) - t_setup
        if denominator > 1e-6:
            sat.rho = min(1.0, (sat.t_train_global - t_setup) / denominator)
        else:
            sat.rho = 1.0

        sat.n_train = math.floor(sat.rho * sat.n_train)


def train_local_model(
    sat: "Satellite",
    device: torch.device = Config.DEVICE,
    epochs: int = Config.LOCAL_TRAIN_EPOCHS,
    lr: float = Config.LOCAL_TRAIN_LR,
):
    """Run local training on a single satellite."""
    sat.local_model = sat.local_model.to(device)
    sat.local_model.train()

    if hasattr(sat.local_model, "set_geo_state"):
        lc_vec = [sat.collected_landcover.get(cat, 0.0) for cat in Config.LANDCOVER_CATEGORIES]
        sat.local_model.set_geo_state(lc_vec)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, sat.local_model.parameters()),
        lr=lr,
    )

    sat.local_train_size = 0
    for x, y in sat.training_dataloader:
        sat.local_train_size += x.shape[0]

    for epoch in range(max(1, int(epochs))):
        for x, y in sat.training_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = sat.local_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    sat.local_model = sat.local_model.to("cpu")
    sat.local_model_round = sat.local_model_round + 1


def receive_parameters(
    satellites: List["Satellite"],
    server_index: int = Config.SINGLE_SERVER_INDEX,
) -> Tuple[List[int], List[float], List[nn.Module]]:
    """Collect local models and normalized sample weights for server aggregation."""
    server_sat = satellites[server_index]
    server_sat.role = "SERVER"
    server_sat.cluster_head = server_index
    server_sat.cluster_members = list(range(len(satellites)))

    uploaded_ids: List[int] = []
    uploaded_weights: List[float] = []
    uploaded_models: List[nn.Module] = []
    tot_samples = 0

    for sat in satellites:
        tot_samples += sat.local_train_size
        uploaded_ids.append(sat.global_index)
        uploaded_weights.append(sat.local_train_size)
        uploaded_models.append(sat.local_model)

    if tot_samples <= 0:
        raise ValueError(f"Total sample count must be greater than 0, current value is {tot_samples}")

    for i, w in enumerate(uploaded_weights):
        uploaded_weights[i] = w / tot_samples

    return uploaded_ids, uploaded_weights, uploaded_models


def aggregate_parameters(
    uploaded_models: List[nn.Module],
    uploaded_weights: List[float],
    server_sat: "Satellite",
) -> None:
    """Aggregate federated parameters on the server using weighted averaging."""
    if not uploaded_models:
        raise ValueError("uploaded_models cannot be empty")
    if len(uploaded_models) != len(uploaded_weights):
        raise ValueError("uploaded_models and uploaded_weights must have the same length")

    server_sat.global_model = copy.deepcopy(uploaded_models[0])

    for name, param in server_sat.global_model.named_parameters():
        if any(name.startswith(prefix) for prefix in FEDERATED_PREFIXES):
            param.data.zero_()

    for w, local_model in zip(uploaded_weights, uploaded_models):
        for name, param in local_model.named_parameters():
            if any(name.startswith(prefix) for prefix in FEDERATED_PREFIXES):
                server_sat.global_model.state_dict()[name].data += param.data.clone() * w


def send_parameters(
    satellites: List["Satellite"],
    aggregated_model: nn.Module,
    server_index: int = Config.SINGLE_SERVER_INDEX,
):
    """Broadcast aggregated federated parameters to all satellites."""
    server_sat = satellites[server_index]

    for sat in satellites:
        for name, param in aggregated_model.named_parameters():
            if any(name.startswith(prefix) for prefix in FEDERATED_PREFIXES):
                sat.local_model.state_dict()[name].data.copy_(param.data)

        if sat is not server_sat:
            sat.role = "WORKER"
            sat.cluster_head = server_index
            sat.cluster_members = []


def main(time_step=300, max_time=6000):
    """Run the main FedAD simulation loop."""
    # ====================================================================
    # ====================================================================
    print("\n" + "="*50 + "Dataset details" + "="*50)

    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"Dataset folder does not exist: {Config.DATA_PATH}")

    full_dataset_raw = datasets.ImageFolder(root=Config.DATA_PATH, transform=None)
    all_targets_raw = np.asarray([sample[1] for sample in full_dataset_raw.samples], dtype=int)
    print(f"Raw dataset size: {len(full_dataset_raw)}")
    class_counts = [f"Label {class_idx}: {len(np.where(all_targets_raw == class_idx)[0])} samples" for class_idx in range(Config.NUM_CLASSES)]
    print(f"Raw samples per label: {', '.join(class_counts)}")
    if len(full_dataset_raw) == 0:
        raise ValueError(f"Dataset is empty; no samples were loaded from: {Config.DATA_PATH}")

    training_data = load_training_data()
    testing_data = load_testing_data()

    print(f"Landcover resolution: {Config.LANDCOVER_RESOLUTION} degrees")
    
    train_dataset, train_class_indices = training_data
    print(f"Training set size: {len(train_dataset)}")
    train_class_counts = [f"Label {class_idx}: {len(train_class_indices[class_idx])} samples" for class_idx in range(Config.NUM_CLASSES)]
    print(f"Training samples per label: {', '.join(train_class_counts)}")

    test_dataset, test_class_indices = testing_data
    print(f"Test set size: {len(test_dataset)}")
    test_class_counts = [f"Label {class_idx}: {len(test_class_indices[class_idx])} samples" for class_idx in range(Config.NUM_CLASSES)]
    print(f"Test samples per label: {', '.join(test_class_counts)}")
    print("="*100 + "\n")

    # ====================================================================
    # ====================================================================
    print("\n" + "="*50 + "Constellation parameters" + "="*50)
    print(f"Orbital inclination: {math.degrees(Config.I):.2f}°")
    print(f"Total satellites: {Config.T}")
    print(f"Orbital planes: {Config.P}")
    print(f"Satellites per plane: {Config.S}")
    print(f"Phase factor: {Config.F}")
    print(f"Orbital altitude: {Config.H/1000:.0f} km")
    print(f"Satellite orbital period: {Config.ORBITAL_PERIOD/60:.4f} minutes = {Config.ORBITAL_PERIOD:.0f} seconds")
    print(f"Time step: {time_step} seconds = {time_step/60:.2f} minutes")
    print(f"Total simulation time: {max_time} seconds = {max_time/60:.2f} minutes")
    print(f"Expected orbital periods: {max_time/Config.ORBITAL_PERIOD:.4f}")

    print(f"\n--- GEO-HA (FedAD) Configuration ---")
    print(f"Model: MAC_ResNet18_FedADL (Geo-Conditioned Hyper-Adapter)")
    print(f"Ablation variants: {MODEL_ABLATION_CONFIG['ablation_variant']}")
    print(f"Feature dimension: {FEDAD_DIM}")
    print(f"LoRA rank: {FEDAD_ADAPTER_RANK}")
    print(f"Hypernet hidden width: {FEDAD_HYPER_HIDDEN}")
    print(f"Geo input dimension: {MODEL_ABLATION_CONFIG['geo_in_dim']} "
          f"(= landcover_dim {MODEL_ABLATION_CONFIG['landcover_dim']})")
    print(f"Adapter enabled: {MODEL_ABLATION_CONFIG['use_adapter']}")
    print(f"Geo conditioning enabled: {MODEL_ABLATION_CONFIG['use_geo']}")
    print(f"Upload mode (UPLOAD_MODE): {UPLOAD_MODE}")
    print(f"Federated aggregation prefixes: {FEDERATED_PREFIXES}")
    print(f"Algorithm name: {Config.ALGORITHM_NAME}")

    satellites = initialize_constellation()

    print("\n" + "="*50 + "Post-initialization device check" + "="*50)
    print(f"Config.DEVICE: {Config.DEVICE}")
    print(f"Satellite 0 initial model device: {next(satellites[0].local_model.parameters()).device}")
    for sat in satellites:
        sat.local_model = sat.local_model.to(Config.DEVICE)
    print(f"Satellite 0 device after conversion: {next(satellites[0].local_model.parameters()).device}")

    print(f"Satellite 0 model type: {type(satellites[0].local_model).__name__}")
    print("="*100 + "\n")

    # ====================================================================
    # ====================================================================
    training_loss_history = []            # Training loss per round
    training_accuracy_history = []        # Training accuracy per round
    full_testing_loss_history = []        # Full test loss per round
    full_testing_accuracy_history = []    # Full test accuracy per round
    past_testing_loss_history = []        # Past-data test loss per round
    past_testing_accuracy_history = []    # Past-data test accuracy per round
    current_testing_loss_history = []     # Current-data test loss per round
    current_testing_accuracy_history = [] # Current-data test accuracy per round
    temporal_knowledge_retention_history = []  # Temporal knowledge retention metric

    communication_round = 0  # Federated communication round counter

    # ====================================================================
    # ====================================================================
    print("\n\n" + "="*50 + "Start simulation" + "="*50)
    print(f"Report positions every {time_step} seconds")

    t = 0
    while t <= max_time:

        theta_G_t = (Config.THETA_G0 + Config.OMEGA_E * t) % (2 * math.pi)

        for sat in satellites:
            sat.update_position(t, theta_G_t)
            if sat.t_training_trigger > 0:
                dist = get_landcover_distribution(sat.lat_rad, sat.lon_rad)
                for cat in Config.LANDCOVER_CATEGORIES:
                    sat.collected_landcover[cat] += dist[cat]

        for i, sat in enumerate(satellites):
            for neighbor_idx, _, _, direction in sat.neighbors:
                neighbor_sat = satellites[neighbor_idx]
                sat.peak_rate[direction] = sat.calculate_communication_link_to(neighbor_sat)
                sat.actual_rate[direction] = sat.peak_rate[direction]

        for i, sat in enumerate(satellites):
            sat.calculate_gpu()

        all_idle = all(sat.t_training_trigger < 0 for sat in satellites)

        if all_idle:
            for i, sat in enumerate(satellites):
                sat.t_train = sat.predict_training_time(None)
            run_consensus_and_downsampling(satellites)
            for sat in satellites:
                sat.t_training_trigger = t + sat.t_train_global
                sat.collected_landcover = {cat: 0.0 for cat in Config.LANDCOVER_CATEGORIES}

        eligible_sats = select_clients(satellites, t)

        if eligible_sats:
            # ============================================================
            # ============================================================
            print("\n" + "-"*50 + f"Time: {t} seconds ({t/60:.2f} minutes) | Communication round: {communication_round}" + "-"*50)

            for sat in tqdm(eligible_sats, desc="[Assign Satellite Data]", unit="sat"):
                normalized_dist = _normalize_probs_dict(sat.collected_landcover)
                assign_dataset_to_satellite(sat, training_data, testing_data, landcover_probs=normalized_dist, seed=sat.global_index + t)

            training_loss, training_accuracy = global_evaluation_and_recording(eligible_sats, mode="train",
                                                                    loss_history=training_loss_history, accuracy_history=training_accuracy_history,
                                                                    desc="[Evaluate Training Data]")
            if training_accuracy > 0:
                print(f"    Current training loss: {training_loss:.4f}, Current training accuracy: {training_accuracy:.4f}")

            full_testing_loss, full_testing_accuracy = global_evaluation_and_recording(eligible_sats, mode="full_test",
                                                                            loss_history=full_testing_loss_history, accuracy_history=full_testing_accuracy_history,
                                                                            desc="[Evaluate Full Test Data]"
                                                                            )
            if full_testing_accuracy > 0:
                print(f"    Full test loss: {full_testing_loss:.4f}, Full test accuracy: {full_testing_accuracy:.4f}")

            if communication_round % 5 == 1 and communication_round > 1:
                past_testing_loss, past_testing_accuracy = global_evaluation_and_recording(eligible_sats, mode="past_test",
                                                                                loss_history=past_testing_loss_history, accuracy_history=past_testing_accuracy_history,
                                                                                desc="[Evaluate Past Test Data]"
                                                                                )
                if past_testing_accuracy > 0:
                    numerator = past_testing_accuracy
                    denominator = sum(current_testing_accuracy_history[1:]) / len(current_testing_accuracy_history[1:])
                    temporal_knowledge_retention = numerator / denominator if denominator > 0 else 0
                    temporal_knowledge_retention_history.append(temporal_knowledge_retention)
                    print(f"    Past test loss: {past_testing_loss:.4f}, Past test accuracy: {past_testing_accuracy:.4f}, Temporal knowledge retention: {temporal_knowledge_retention:.4f}")

            current_testing_loss, current_testing_accuracy = global_evaluation_and_recording(eligible_sats, mode="current_test",
                                                                                  loss_history=current_testing_loss_history, accuracy_history=current_testing_accuracy_history,
                                                                                  desc="[Evaluate Current Test Data]"
                                                                                  )
            if current_testing_accuracy > 0:
                print(f"    Current test loss: {current_testing_loss:.4f}, Current test accuracy: {current_testing_accuracy:.4f}")

            for sat in tqdm(eligible_sats, desc="[Local training]", unit="sat"):
                train_local_model(sat, device=Config.DEVICE, epochs=Config.LOCAL_TRAIN_EPOCHS, lr=Config.LOCAL_TRAIN_LR)

            uploaded_ids, uploaded_weights, uploaded_models = receive_parameters(satellites, Config.SINGLE_SERVER_INDEX)
            server_sat = satellites[Config.SINGLE_SERVER_INDEX]

            aggregate_parameters(uploaded_models, uploaded_weights, server_sat)

            send_parameters(satellites, server_sat.global_model, Config.SINGLE_SERVER_INDEX)

            communication_round += 1
            for sat in eligible_sats:
                if sat.current_testing_indices:
                    existing = set(sat.past_testing_indices)
                    new_indices = [idx for idx in sat.current_testing_indices if idx not in existing]
                    if new_indices:
                        lc_now = [sat.collected_landcover.get(cat, 0.0) for cat in Config.LANDCOVER_CATEGORIES]
                        s_now = compute_geo_state_tensor(lc_now)
                        sat.past_testing_indices.extend(new_indices)
                        sat.past_testing_geo_states.extend([s_now] * len(new_indices))

                sat.t_training_trigger = -1.0

        t += time_step

    # ====================================================================
    # ====================================================================
    save_results_to_csv(full_testing_accuracy_history, past_testing_accuracy_history,
                        temporal_knowledge_retention_history, current_testing_accuracy_history,
                        Config.ALGORITHM_NAME, Config.DATASET_NAME
                        )


if __name__ == "__main__":
    for run in range(5):
        print(f"\n{'='*100}")
        print(f"{'='*40} Start run {run+1}/5 {'='*40}")
        print(f"{'='*100}\n")
        main(time_step=1, max_time=Config.ORBITAL_PERIOD+1)
        print(f"\n{'='*100}")
        print(f"{'='*40} Run {run+1}/5 completed {'='*40}")
        print(f"{'='*100}\n")
