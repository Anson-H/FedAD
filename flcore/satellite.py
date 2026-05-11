#!/usr/bin/env python3
"""
Satellite model and constellation utilities.
"""

import copy
import math
from typing import Dict, List

from flcore.config import Config


class Satellite:
    """Store orbital, communication, energy, dataset, and model state for one satellite."""

    def __init__(self, plane_index: int, sat_index: int, omega_p: float, u_ps_0: float):
        # Orbital parameters
        self.plane_index = plane_index
        self.sat_index = sat_index
        self.global_index = plane_index * Config.S + sat_index
        self.omega_p = omega_p
        self.u_ps_0 = u_ps_0
        self.neighbors = self.get_neighbor_indices()

        # Position state
        self.u_ps_t = 0.0
        self.lat_rad = 0.0
        self.lon_rad = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.neighbor_d_straight = {}
        self.neighbor_d_spherical = {}

        # Eclipse and communication state
        self.in_sunlight = True
        self.eclipse_status = ""
        self.traffic_demand = 0
        self.traffic_demand_neighbor_provided = {}
        self.peak_rate = {}
        self.actual_rate = {}
        self.P_T = 0.0
        self.P_R = 0.0
        self.P_T_ELEC = 0.0
        self.P_R_ELEC = 0.0

        # Compute state
        self.P_CPU = Config.P_CPU
        self.f_CPU = Config.f_CPU
        self.P_GPU = 0.0
        self.f_GPU = 0.0
        self.flops_GPU = 0.0

        # Energy state
        self.max_capacity = Config.DoD * Config.ENERGY_MAX
        self.current_capacity = self.max_capacity
        self.R_charge = 0.0
        self.P_charge = 0.0
        self.P_consumption = 0.0
        self.P_BASE = Config.P_BASE
        self.P_trans_elec = 0.0
        self.time_to_eclipse_exit = None
        self.time_to_eclipse_enter = None
        self.energy_gain_aggregation = 0.0
        self.capacity_after_aggregation = self.current_capacity
        self.capacity_delta_aggregation = 0.0

        # Dataset state
        self.training_data = None
        self.training_dataloader = None
        self.exemplar_data = None
        self.exemplar_dataloader = None
        self.training_combined_data = None
        self.training_combined_dataloader = None
        self.full_testing_data = None
        self.full_testing_dataloader = None
        self.current_testing_indices = []
        self.current_testing_data = None
        self.current_testing_dataloader = None
        self.past_testing_indices = []
        self.past_testing_data = None
        self.past_testing_dataloader = None
        self.past_testing_geo_states: List = []

        # Federated learning model state
        self.local_model = copy.deepcopy(Config.MODEL)
        self.local_model_round = 0
        self.global_model = None
        self.global_model_round = 0
        self.global_base_model = None
        self.prev_model = None

        # Optional algorithm-specific model components
        self.global_extractor = None
        self.local_extractor = None
        self.local_head = None
        self.gating_network = None
        self.model_old = [None, None]

        # Training and evaluation metrics
        self.local_train_loss = 0.0
        self.local_train_acc = 0.0
        self.local_train_size = 0.0
        self.local_full_testing_loss = 0.0
        self.local_full_testing_acc = 0.0
        self.local_full_testing_size = 0.0
        self.local_past_testing_loss = 0.0
        self.local_past_testing_acc = 0.0
        self.local_past_testing_size = 0.0
        self.local_current_testing_loss = 0.0
        self.local_current_testing_acc = 0.0
        self.local_current_testing_size = 0.0

        # Transmission and training schedule state
        self.transmission_round = 0
        self.is_transmitting = True
        self.transmission_remaining_bits = {
            "forward": Config.S_W,
            "backward": Config.S_W,
            "left": Config.S_W,
            "right": Config.S_W,
        }
        self.transmission_accumulated_time = {
            "forward": 0.0,
            "backward": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
        self.transmission_remaining_time = {
            "forward": 0.0,
            "backward": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
        self.t_train = 0.0
        self.e_train = 0.0
        self.t_aggregation = 0.0
        self.e_aggregation = 0.0
        self.t_train_global = 0.0
        self.rho = 1.0
        self.n_train = Config.D_K
        self.S_COMP = 0.0
        self.S_C = 0.0
        self.S_E = 0.0
        self.role = "WORKER"
        self.cluster_head = None
        self.cluster_members = []
        self.t_training_trigger = -1.0

        # Landcover and dataset distribution summaries
        self.collected_landcover = {cat: 0.0 for cat in Config.LANDCOVER_CATEGORIES}
        self.train_dist_str = ""
        self.full_test_dist_str = ""
        self.current_test_dist_str = ""
        self.past_test_dist_str = ""

        # Optional regularization, prototype, and exemplar state
        self.fisher_matrix = None
        self.data_protos = None
        self.protos = None
        self.global_protos = [None for _ in range(Config.NUM_CLASSES)]
        self.V = 0.0
        self.h = None
        self.exemplar_indices = {i: [] for i in range(Config.NUM_CLASSES)}

    def get_neighbor_indices(self) -> List:
        """Return the four ring-grid neighbors for this satellite."""

        def calc_idx(plane_idx, sat_idx):
            return plane_idx * Config.S + sat_idx

        return [
            (
                calc_idx(self.plane_index, (self.sat_index + 1) % Config.S),
                self.plane_index,
                (self.sat_index + 1) % Config.S,
                "forward",
            ),
            (
                calc_idx(self.plane_index, (self.sat_index - 1 + Config.S) % Config.S),
                self.plane_index,
                (self.sat_index - 1 + Config.S) % Config.S,
                "backward",
            ),
            (
                calc_idx((self.plane_index - 1 + Config.P) % Config.P, self.sat_index),
                (self.plane_index - 1 + Config.P) % Config.P,
                self.sat_index,
                "left",
            ),
            (
                calc_idx((self.plane_index + 1) % Config.P, self.sat_index),
                (self.plane_index + 1) % Config.P,
                self.sat_index,
                "right",
            ),
        ]

    def update_position(self, t: float, theta_G_t: float):
        """Update orbital phase, latitude, longitude, and ECEF coordinates."""
        self.u_ps_t = (self.u_ps_0 + Config.OMEGA_S * t) % (2 * math.pi)
        self.lat_rad = math.asin(math.sin(self.u_ps_t) * math.sin(Config.I_RAD))
        self.lon_rad = (
            self.omega_p
            + math.atan2(math.sin(self.u_ps_t) * math.cos(Config.I_RAD), math.cos(self.u_ps_t))
            - theta_G_t
        )
        self.lon_rad = self.lon_rad % (2 * math.pi)
        if self.lon_rad > math.pi:
            self.lon_rad -= 2 * math.pi

        r = Config.R + Config.H
        self.x = r * math.cos(self.lat_rad) * math.cos(self.lon_rad)
        self.y = r * math.cos(self.lat_rad) * math.sin(self.lon_rad)
        self.z = r * math.sin(self.lat_rad)

    def calculate_gpu(self):
        """Compute GPU power, frequency, and FLOPs under the always-sunlit assumption."""
        self.P_GPU = Config.P_GPU_MAX
        self.f_GPU = max(0.0, (715 * math.log(self.P_GPU) - 1350) * 1e6)
        self.flops_GPU = Config.ETA_GPU * (Config.N_CUDA * self.f_GPU * 2)

    def predict_training_time(self, constants: Dict[str, float]) -> float:
        """Predict the simplified total local training duration."""
        t_setup = Config.ALPHA_WORKER * Config.T_WORKER
        t_load = (Config.LOCAL_TRAIN_BATCH_SIZE * Config.C_LOAD) / (Config.ETA_CPU * self.f_CPU)
        flops_per_epoch = self.n_train * (Config.C_FORWARD + Config.C_BACKWARD)
        f_gpu_val = max(0.0, (578 * math.log(Config.P_GPU_MAX) - 1555) * 1e6)
        flops_gpu = Config.ETA_GPU * (Config.N_CUDA * f_gpu_val * 2)
        t_compute = flops_per_epoch / (flops_gpu if flops_gpu > 0 else 1e-9)
        t_total = Config.LOCAL_TRAIN_EPOCHS * (t_load + t_compute)
        _ = t_setup
        _ = constants
        return t_total

    def calculate_communication_link_to(self, other_sat: "Satellite") -> float:
        """Compute peak optical communication rate to another satellite."""
        dx = other_sat.x - self.x
        dy = other_sat.y - self.y
        dz = other_sat.z - self.z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

        G_T = 16 / (Config.THETA_T ** 2)
        G_R = (math.pi * Config.D_R / Config.WAVELENGTH) ** 2
        L_T = math.exp(-G_T * Config.THETA_PE ** 2)
        L_R = math.exp(-G_R * Config.THETA_PE ** 2)
        L_FSP = (Config.WAVELENGTH / (4 * math.pi * d)) ** 2
        P_OPT = Config.P_T * G_T * G_R * Config.ETA_T * Config.ETA_R * L_T * L_R * L_FSP
        P_ELEC = (Config.R_P * P_OPT) ** 2

        sigma_s_sq = 2 * Config.Q_ELECTRON * Config.R_P * P_OPT * Config.BANDWIDTH
        sigma_d_sq = 2 * Config.Q_ELECTRON * Config.I_D * Config.BANDWIDTH
        sigma_t_sq = 4 * Config.K_B * Config.T_RNT * Config.BANDWIDTH / Config.R_L
        p_noise = sigma_s_sq + sigma_d_sq + sigma_t_sq
        snr = P_ELEC / p_noise
        return Config.BANDWIDTH * math.log2(1 + snr)


def initialize_constellation() -> List[Satellite]:
    """Create the Walker-Delta satellite constellation and print its topology."""
    satellites = []
    print("\n" + "=" * 50 + "Satellite Parameters" + "=" * 50)

    for p in range(Config.P):
        omega_p = (Config.OMEGA_0 + p * Config.DELTA_OMEGA) % (2 * math.pi)
        for s in range(Config.S):
            u_ps_0 = (Config.U_0_0 + p * Config.DELTA_PSI + s * Config.DELTA_PHI) % (2 * math.pi)
            satellite = Satellite(p, s, omega_p, u_ps_0)
            satellites.append(satellite)

            print(
                f"Satellite ID: {satellite.global_index:4d} | "
                f"Plane index: {satellite.plane_index:2d} | "
                f"In-plane index: {satellite.sat_index:2d} | "
                f"Neighbors: F{satellite.neighbors[0][0]:4d}({satellite.neighbors[0][1]:2d},{satellite.neighbors[0][2]:2d}) "
                f"B{satellite.neighbors[1][0]:4d}({satellite.neighbors[1][1]:2d},{satellite.neighbors[1][2]:2d}) "
                f"L{satellite.neighbors[2][0]:4d}({satellite.neighbors[2][1]:2d},{satellite.neighbors[2][2]:2d}) "
                f"R{satellite.neighbors[3][0]:4d}({satellite.neighbors[3][1]:2d},{satellite.neighbors[3][2]:2d}) | "
                f"RAAN: {math.degrees(satellite.omega_p):7.2f} deg | "
                f"Initial phase: {math.degrees(satellite.u_ps_0):7.2f} deg"
            )

    return satellites
