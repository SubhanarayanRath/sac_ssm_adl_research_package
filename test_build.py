import sys
import os

# Ensure src directory is in Python path
sys.path.append("src")

from tar_data_utils import load_tar_dataset
from models.modular_sac import build_modular_sac

# Configuration matching standard SAC-SSM hyperparameters
cfg = {
    "model": {
        "event_embedding_dim": 32,
        "type_embedding_dim": 16,
        "room_embedding_dim": 16,
        "state_embedding_dim": 16,
        "time_projection_dim": 16,
        "model_dim": 64,
        "dropout": 0.2,
        "ssm_hidden_dim": 64,
        "branch_output_dim": 64
    },
    "data": {
        "short_window": 15,
        "medium_window": 30,
        "long_window": 60
    }
}

print("--- Step 1: Fetching Vocab Config from Pipeline ---")
_, _, _, vocab_config = load_tar_dataset()

print("\n--- Step 2: Instantiating Modular Model (Backbone: GRUwE) ---")
try:
    model = build_modular_sac(
        num_events=vocab_config["num_events"],
        num_types=vocab_config["num_types"],
        num_rooms=vocab_config["num_rooms"],
        num_states=vocab_config["num_states"],
        num_classes=vocab_config["num_classes"],
        num_gap_bins=10,
        max_window=vocab_config["max_window"],
        time_dim=vocab_config["time_dim"],
        context_dim=vocab_config["context_dim"],
        cfg=cfg,
        backbone_type="gruwe"
    )
    
    print("\n--- Step 3: SUCCESS! Model Architecture Summary ---")
    model.summary()
    
except Exception as e:
    print(f"\nERROR: Failed to build model:\n{e}")
