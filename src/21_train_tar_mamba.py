import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from models.backbones.tar_mamba import TARMambaBackbone

def build_tar_mamba_model(seq_len, feature_dim, num_classes):
    """
    Assembles the full TAR-Mamba architecture with multi-task heads.
    """
    inputs = layers.Input(shape=(seq_len, feature_dim), name="sensor_input")

    # 1. Pass through our new backbone
    tar_backbone = TARMambaBackbone(units=64, name="tar_backbone")
    pooled_representation, fused_states, b_t, attention_weights = tar_backbone(inputs)

    # 2. Main Task Head: Activity Classification
    # Uses the transition-aware pooled representation
    x = layers.Dropout(0.3)(pooled_representation)
    activity_out = layers.Dense(num_classes, activation='softmax', name='activity_output')(x)

    # 3. Auxiliary Task Head: Boundary Prediction
    # We output b_t directly. Naming it allows us to attach a specific loss to it.
    # b_t shape is (batch_size, seq_len, 1), already passed through sigmoid.
    boundary_out = layers.Activation('linear', name='boundary_output')(b_t)

    # Build Model
    model = models.Model(
        inputs=inputs,
        outputs=[activity_out, boundary_out],
        name="TAR_Mamba"
    )

    # 4. Compile with Multi-Task Loss
    # We weight the main activity loss higher than the auxiliary boundary loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'activity_output': 'sparse_categorical_crossentropy',
            'boundary_output': 'binary_crossentropy'
        },
        loss_weights={
            'activity_output': 1.0,
            'boundary_output': 0.5  # Tunes how strongly we force the boundary detector
        },
        metrics={
            'activity_output': ['accuracy']
        }
    )

    return model

def main():
    print("="*50)
    print("PHASE 10G: TAR-Mamba Training")
    print("="*50)

    # 1. Define dimensions (Replace with your actual dimensions)
    seq_len = 120
    feature_dim = 45
    num_classes = 12

    # 2. Build Model
    model = build_tar_mamba_model(seq_len, feature_dim, num_classes)
    model.summary()

    # 3. Load Data
    # NOTE: Your data loader needs to return a tuple of inputs and a DICTIONARY of labels
    # e.g., X_train, {'activity_output': y_activity, 'boundary_output': y_boundary}
    # X_train, y_train_dict, X_val, y_val_dict = load_tar_mamba_data()

    # 4. Setup Callbacks
    os.makedirs("models", exist_ok=True)
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath="models/tar_mamba_best.keras",
            monitor="val_activity_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_activity_output_accuracy",
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4
        )
    ]

    # 5. Train
    print("Starting training...")
    # history = model.fit(
    #     X_train, y_train_dict,
    #     validation_data=(X_val, y_val_dict),
    #     epochs=50,
    #     batch_size=64,
    #     callbacks=callbacks_list
    # )
    print("Training pipeline ready.")

if __name__ == "__main__":
    main()
