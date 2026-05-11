#!/usr/bin/env python3
"""
Configuration file.
Contains system-wide constants and experiment parameters.
"""

import math
import os

import torch

from flcore.models import ResNet18


class Config:
    """System configuration class."""

    # ========================================================================
    # Physical constants
    # ========================================================================
    R = 6371e3  # Earth radius (m)
    H = 850e3  # Satellite orbital altitude (m)
    MU = 3.986e14  # Earth standard gravitational parameter (m^3/s^2)
    OMEGA_E = 7.292115e-5  # Earth mean angular velocity (rad/s)
    Q_ELECTRON = 1.6e-19  # Electron charge (C)
    K_B = 1.38e-23  # Boltzmann constant (J/K)

    # ========================================================================
    # Walker-Delta constellation parameters
    # ========================================================================
    I = 50  # Orbital inclination (deg)
    I_RAD = math.radians(I)
    T = 200  # Total number of satellites
    P = 20  # Total number of orbital planes
    S = T // P  # Satellites per orbital plane
    F = 0  # Phasing factor
    DELTA_OMEGA = 2 * math.pi / P  # RAAN gap between planes
    DELTA_PHI = 2 * math.pi / S  # In-plane phase gap
    DELTA_PSI = 2 * math.pi * F / T  # Inter-plane phase gap
    OMEGA_S = math.sqrt(MU / (R + H) ** 3)  # Satellite orbital angular velocity
    ORBITAL_PERIOD = 2 * math.pi / OMEGA_S  # Satellite orbital period (s)

    # ========================================================================
    # Initial conditions
    # ========================================================================
    OMEGA_0 = 0.0  # Reference plane (p=0) RAAN (rad)
    U_0_0 = 0.0  # Reference satellite (p=0, s=0) initial phase (rad)
    THETA_G0 = 0.0  # Initial Greenwich sidereal angle (rad)

    # ========================================================================
    # Sun position parameters
    # ========================================================================
    LAMBDA_SUN = math.radians(90)  # Sun right ascension (rad)
    PHI_SUN = math.radians(23)  # Sun declination (rad)

    # ========================================================================
    # Compute system parameters
    # ========================================================================
    P_CPU = 20  # CPU power consumption (W)
    f_CPU = 3.2e9  # CPU frequency (Hz)
    ETA_CPU = 0.9  # CPU utilization

    P_GPU_MAX = 70  # GPU peak power consumption (W)
    N_CUDA = 2560  # CUDA cores
    N_TENSOR = 96  # Tensor cores
    ETA_GPU = 0.3  # GPU utilization

    # ========================================================================
    # Training simulation parameters
    # ========================================================================
    ALPHA_WORKER = 2  # Number of worker processes
    T_WORKER = 6.0  # Startup time per process (s)
    C_LOAD = 124.8e6  # Data preprocessing cycles
    D_K = 5000  # Local sample count
    C_FORWARD = 8.2e9  # Forward FLOPs per sample
    C_BACKWARD = 16.4e9  # Backward FLOPs per sample

    # ========================================================================
    # Aggregation simulation parameters
    # ========================================================================
    W_MODEL = 23528522  # Number of model parameters (ResNet-50 reference)
    ALPHA_BYTES = 4  # Bytes per parameter (FP32)
    BW_PCIE = 32e9  # PCIe bandwidth (32 GB/s)
    C_AGGREGATE = 1  # FLOPs needed per parameter during aggregation
    ALPHA_MEM = 2  # Extra memory overhead factor
    BW_GPU_MEM = 1008e9  # GPU memory bandwidth (3090Ti, 1008 GB/s)

    # ========================================================================
    # Training prediction parameters
    # ========================================================================
    T_MIN_LIMIT = 5.0  # Minimum allowed training duration (s)

    # ========================================================================
    # Communication system parameters
    # ========================================================================
    WAVELENGTH = 1550e-9  # Signal wavelength (m)
    C_LIGHT = 3e8  # Speed of light (m/s)
    F_C = C_LIGHT / WAVELENGTH  # Carrier frequency (Hz)
    P_T = 0.7  # Transmit optical power (W)
    BANDWIDTH = 1e9  # Channel bandwidth (Hz)

    # ========================================================================
    # Communication device parameters
    # ========================================================================
    R_P = 0.6  # Photodetector responsivity (A/W)
    ETA_T = 0.8  # Transmitter optical efficiency
    ETA_R = 0.8  # Receiver optical efficiency
    THETA_T = 15e-6  # Full transmit divergence angle (rad)
    D_R = 0.135  # Receiver aperture diameter (m)
    THETA_PE = 1e-6  # Pointing error angle (rad)
    I_D = 1e-9  # Dark current (A)
    T_RNT = 500  # Receiver noise temperature (K)
    R_L = 10e3  # Load resistance (ohm)
    ETA_EO = 0.2  # Electro-optical conversion efficiency

    # ========================================================================
    # Transmission parameters
    # ========================================================================
    S_W = 1000 * 1024 * 1024 * 8  # Model size (bits) - 100 MB

    # ========================================================================
    # Power/endurance parameters
    # ========================================================================
    ETA_CHARGE = 0.19  # Energy conversion efficiency
    G_SC = 1368  # Solar irradiance (W/m^2)
    S_S = 30  # Solar panel area (m^2)
    P_BASE = 100  # Baseline power consumption (W)
    CAPACITY_MAX = 51 * 3600  # Maximum battery capacity (A*s)
    ENERGY_MAX = 1685 * 3600  # Maximum energy (W*s)
    VOLTAGE = ENERGY_MAX / CAPACITY_MAX  # Voltage (V)
    VOLTAGE_MIN = 22.5  # Minimum operating voltage (V)
    VOLTAGE_MAX = 37.8  # Maximum operating voltage (V)
    I_MAX = 15  # Maximum current (A)
    DoD = 0.1  # Depth-of-discharge limit
    ENERGY_ESTIMATION_MAX_STEP = 1.0  # Max simulation step for charge estimation (s)

    # ========================================================================
    # Election and clustering parameters
    # ========================================================================
    SINGLE_SERVER_INDEX = 0  # Single constellation-wide server satellite

    # ========================================================================
    # Default local training parameters
    # ========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10
    MODEL = None
    DATA_PATH = None
    LANDCOVER_TO_LABELS = None
    LOCAL_TRAIN_EPOCHS = 1
    LOCAL_TRAIN_BATCH_SIZE = 16
    LOCAL_TRAIN_LR = 1e-3
    LOCAL_TEST_BATCH_SIZE = 64

    # ========================================================================
    # Experiment configuration parameters
    # ========================================================================
    ALGORITHM_NAME = "FedAvg"
    DATASET_NAME = "NWPU"
    MU = 0.01
    LAMDA = 1.0

    # ========================================================================
    # Dataset-related constants
    # ========================================================================
    LANDCOVER_CATEGORIES = [
        "Tree cover",
        "Shrubland",
        "Grassland",
        "Cropland",
        "Built-up",
        "Bare / sparse vegetation",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Mangroves",
        "Moss and lichen",
    ]
    LANDCOVER_RESOLUTION = 3  # Degrees; supported values include 3 and 15
    DATASET_ASSIGNMENT_SEED_BASE = 0

    EUROSAT_DATA_PATH = os.path.join("data", "eurosat", "2750")
    EUROSAT_TRAIN_RATIO = 0.9
    LANDCOVER_TO_EUROSAT_LABELS = {
        "Tree cover": [1],
        "Shrubland": [2],
        "Grassland": [5],
        "Cropland": [0, 6],
        "Built-up": [3, 4, 7],
        "Bare / sparse vegetation": [2],
        "Snow and ice": [8, 9],
        "Permanent water bodies": [8, 9],
        "Herbaceous wetland": [2],
        "Mangroves": [1],
        "Moss and lichen": [2],
    }

    NWPU_DATA_PATH = os.path.join("data", "NWPU-RESISC45", "NWPU-RESISC45")
    NWPU_TRAIN_RATIO = 0.9
    LANDCOVER_TO_NWPU_LABELS = {
        "Tree cover": [13],
        "Shrubland": [6],
        "Grassland": [15, 16, 22],
        "Cropland": [8, 31, 42],
        "Built-up": [
            0, 1, 2, 3, 5, 7, 10, 11, 14, 17, 18, 19, 23,
            24, 26, 27, 28, 29, 30, 33, 34, 38, 39, 40, 41, 43,
        ],
        "Bare / sparse vegetation": [4, 6, 12, 25],
        "Snow and ice": [9, 35, 37],
        "Permanent water bodies": [20, 21, 32, 36],
        "Herbaceous wetland": [44],
        "Mangroves": [13],
        "Moss and lichen": [6],
    }

    @classmethod
    def set_dataset(cls, name: str):
        """Switch dataset-dependent settings."""
        cls.DATASET_NAME = name
        if name == "EuroSAT":
            cls.NUM_CLASSES = 10
            cls.DATA_PATH = cls.EUROSAT_DATA_PATH
            cls.LANDCOVER_TO_LABELS = cls.LANDCOVER_TO_EUROSAT_LABELS
            cls.LANDCOVER_RESOLUTION = 3
        elif name == "NWPU":
            cls.NUM_CLASSES = 45
            cls.DATA_PATH = cls.NWPU_DATA_PATH
            cls.LANDCOVER_TO_LABELS = cls.LANDCOVER_TO_NWPU_LABELS
            cls.LANDCOVER_RESOLUTION = 3
        else:
            raise ValueError(
                f"Unsupported dataset: {name}, valid options are 'EuroSAT' or 'NWPU'"
            )
        cls.MODEL = ResNet18(in_features=3, num_classes=cls.NUM_CLASSES)

    # ========================================================================
    # Global Internet user distribution (million users)
    # ========================================================================
    # Grid resolution: 12 rows (15 degrees latitude) x 24 columns (15 degrees longitude)
    INTERNET_USERS_DISTRIBUTION = [
        [0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [5, 17, 19, 2, 2, 2, 2, 2, 1, 1, 1, 1, 4, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [1, 17, 0, 19, 19, 19, 19, 19, 2, 0, 0, 46, 178, 76, 89, 89, 38, 38, 16, 0, 6, 0, 5, 0],
        [1, 0, 0, 17, 32, 17, 17, 17, 0, 0, 1, 51, 52, 79, 71, 27, 41, 70, 67, 134, 151, 38, 0, 0],
        [0, 17, 0, 0, 15, 33, 38, 24, 0, 0, 1, 13, 3, 15, 22, 37, 115, 116, 118, 156, 149, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 26, 24, 15, 0, 0, 13, 99, 5, 27, 1, 1, 94, 40, 39, 34, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 10, 31, 14, 14, 0, 1, 2, 5, 31, 1, 0, 0, 10, 10, 13, 13, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 18, 21, 14, 0, 1, 1, 15, 14, 1, 0, 0, 0, 2, 2, 2, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4],
        [0, 0, 0, 0, 0, 0, 0, 10, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
