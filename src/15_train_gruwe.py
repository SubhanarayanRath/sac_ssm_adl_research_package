import os
import sys
import yaml
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, losses, metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tar_data_utils import load_tar_dataset
from src.models.modular_sac import build_modular_sac

print("===================================================")
print(" PHASE 7: Training GRUwE Baseline (YAML-Configured)")
print("===================================================")

# Optimized hyperparameters for RTX 4050 (6GB)
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.0008

def main():
    # 1. Load Centralized Configuration from YAML
    config_path = "config/default.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Cannot find configuration file at {config_path}")
        return

    print(f"Loading base configuration from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load Data
    train_ds, val_ds, test_ds, vocab_config = load_tar_dataset(
        filepath="data/processed/aruba_tar_windows.npz",
        batch_size=BATCH_SIZE
    )

    # Map dataset to yield (inputs, activity_target) directly
    def adapt_ds(ds):
        return ds.map(lambda x, y: (x, y["activity"]))

    train_ds_mapped = adapt_ds(train_ds)
    val_ds_mapped = adapt_ds(val_ds)

    print("\nBuilding modular model with 'gruwe' backbone (Single Output)...")
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
        backbone_type="gruwe",
        auxiliary_tasks=False
    )

    # Compile with standard classification loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    model.summary()

    # Print exact parameter count for paper comparison table
    print(f"\n[PAPER METRIC] Total Trainable Parameters (GRUwE): {model.count_params():,}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/gruwe_baseline.keras"

    cbs = [
        callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n--- Starting GRUwE Baseline Training on RTX 4050 ---")
    history = model.fit(
        train_ds_mapped,
        validation_data=val_ds_mapped,
        epochs=EPOCHS,
        callbacks=cbs
    )

    print(f"\nSUCCESS: Training complete. Best model saved to {model_path}")

if __name__ == "__main__":
    main()