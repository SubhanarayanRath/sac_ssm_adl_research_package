from .irregular_gru import IrregularTimeGRU
from .selective_mamba import SelectiveMamba
from .momentum_mamba import MomentumMamba
from .tar_mamba_v1 import TARMambaV1Backbone

BACKBONES = {
    "gruwe": IrregularTimeGRU,
    "mamba": SelectiveMamba,
    "momentum_mamba": MomentumMamba,
    "tar_mamba": TARMambaV1Backbone,
}

def get_backbone(name: str, units: int, dropout: float = 0.0, return_sequences: bool = True, name_prefix: str = ""):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: '{name}'. Available: {list(BACKBONES.keys())}")

    return BACKBONES[name](
        units=units,
        dropout=dropout,
        return_sequences=return_sequences,
        name=f"{name_prefix}_{name}"
    )