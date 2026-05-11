import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(in_features, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def compute_geo_state_tensor(landcover_vec) -> torch.Tensor:
    """Convert a landcover distribution into a normalized Hellinger geo-state vector."""
    if torch.is_tensor(landcover_vec):
        vec = landcover_vec.detach().float().flatten().cpu()
    else:
        vec = torch.tensor(list(landcover_vec), dtype=torch.float32)

    total = vec.sum()
    if total > 1e-8:
        vec = vec / total
    vec = vec.clamp(min=0.0).sqrt()

    return vec


class GeoConditionedAdapter(nn.Module):
    """LoRA-style adapter modulated by a geographic state vector."""

    def __init__(
        self,
        dim: int = 512,
        rank: int = 32,
        geo_in_dim: int = 11,
        hyper_hidden: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.geo_in_dim = geo_in_dim

        self.W_down = nn.Linear(dim, rank, bias=False)
        self.W_up = nn.Linear(rank, dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(dim))

        self.hyper = nn.Sequential(
            nn.Linear(geo_in_dim, hyper_hidden),
            nn.GELU(),
            nn.Linear(hyper_hidden, 2 * rank),
        )

        self._init_identity()

    def _init_identity(self):
        """Initialize the adapter so it starts as an identity residual path."""
        nn.init.zeros_(self.W_up.weight)
        nn.init.zeros_(self.hyper[-1].weight)
        self.hyper[-1].bias.data[: self.rank].fill_(1.0)
        self.hyper[-1].bias.data[self.rank :].zero_()

    def forward(self, z_shared: torch.Tensor, geo_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_shared:  [B, dim]
            geo_state: [geo_in_dim] (per-satellite per-round) [B, geo_in_dim]
        Returns:
            delta:     [B, dim]
        """
        if geo_state.dim() == 1:
            geo_state = geo_state.unsqueeze(0)  # [1, geo_in_dim]

        gamma_beta = self.hyper(geo_state)           # [G, 2*rank]
        gamma, beta = gamma_beta.chunk(2, dim=-1)    # [G, rank]

        z_down = self.W_down(z_shared)               # [B, rank]
        z_mod = gamma * z_down + beta                # [B, rank]
        z_mod = F.gelu(z_mod)

        delta = self.W_up(z_mod) + self.bias         # [B, dim]
        return delta


class MAC_ResNet18_FedAD(nn.Module):
    """Ground-track-aware federated model with a geo-conditioned adapter."""

    ABLATION_VARIANTS = {
        "full":   {"use_adapter": True,  "use_geo": True},
        "wo_geo": {"use_adapter": True,  "use_geo": False},
        "wo_ad": {"use_adapter": False, "use_geo": False},
    }

    def __init__(
        self,
        in_features: int = 3,
        num_classes: int = 10,
        dim: int = 512,
        adapter_rank: int = 32,
        hyper_hidden: int = 64,
        landcover_dim: int = 11,
        ablation_variant: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.adapter_rank = adapter_rank
        self.landcover_dim = landcover_dim
        self.geo_in_dim = landcover_dim

        variant_key = (ablation_variant or "full").lower()
        if variant_key not in self.ABLATION_VARIANTS:
            supported = ", ".join(sorted(self.ABLATION_VARIANTS.keys()))
            raise ValueError(
                f"Unsupported ablation_variant: {ablation_variant}. Value: {supported}"
            )
        flags = self.ABLATION_VARIANTS[variant_key]
        self.ablation_variant = variant_key
        self.use_adapter = flags["use_adapter"]
        self.use_geo = flags["use_geo"]

        resnet_shared = models.resnet18(weights=None)
        resnet_shared.conv1 = nn.Conv2d(
            in_features, 64, kernel_size=5, stride=1, padding=2, bias=False
        )
        resnet_shared.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.shared_encoder = nn.Sequential(*list(resnet_shared.children())[:-1])

        # === Geo-Conditioned Adapter ===
        if self.use_adapter:
            self.geo_adapter = GeoConditionedAdapter(
                dim=dim,
                rank=adapter_rank,
                geo_in_dim=self.geo_in_dim,
                hyper_hidden=hyper_hidden,
            )
        else:
            self.geo_adapter = None

        self.head = nn.Linear(dim, num_classes)

        self.register_buffer("current_geo_state", torch.zeros(self.geo_in_dim))

    @torch.no_grad()
    def set_geo_state(self, landcover_vec) -> None:
        """Store the current satellite landcover distribution as the model geo state."""
        if torch.is_tensor(landcover_vec):
            vec = landcover_vec.detach().float().flatten()
        else:
            vec = torch.tensor(list(landcover_vec), dtype=torch.float32)

        if vec.numel() != self.landcover_dim:
            raise ValueError(
                f"landcover_vec dimension mismatch: expected {self.landcover_dim}, {vec.numel()}"
            )

        total = vec.sum()
        if total > 1e-8:
            vec = vec / total
        vec = vec.clamp(min=0.0).sqrt()

        self.current_geo_state.copy_(vec.to(self.current_geo_state.device))

    def get_ablation_config(self):
        return {
            "ablation_variant": self.ablation_variant,
            "use_adapter": self.use_adapter,
            "use_geo": self.use_geo,
            "adapter_rank": self.adapter_rank,
            "landcover_dim": self.landcover_dim,
            "geo_in_dim": self.geo_in_dim,
        }

    def forward(self, x, geo_state=None):
        if geo_state is None:
            geo_state = self.current_geo_state
        else:
            if not torch.is_tensor(geo_state):
                geo_state = torch.as_tensor(geo_state, dtype=torch.float32)
            geo_state = geo_state.to(device=x.device, dtype=torch.float32)

        # ---------------- Shared Trunk ----------------
        z_shared = self.shared_encoder(x).flatten(1)  # [B, dim]

        # ---------------- Geo-Conditioned Adapter ----------------
        if self.use_adapter and self.geo_adapter is not None:
            if self.use_geo:
                effective_geo = geo_state
            else:
                effective_geo = torch.zeros_like(geo_state)

            delta = self.geo_adapter(z_shared, effective_geo)  # [B, dim]
            z_final = z_shared + delta  # single-skip fusion
        else:
            z_final = z_shared

        logits = self.head(z_final)
        return logits
