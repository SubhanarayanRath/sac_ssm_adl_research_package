import tensorflow as tf
from tensorflow.keras import layers
from src.models.backbones.tar_mamba_v1 import TARMambaV1Backbone
from src.models.layers import AttentionPooling, WeightedScaleFusion

def build_modular_sac(num_events, num_types, num_rooms, num_states, num_classes, max_window, time_dim, context_dim, cfg, backbone_type="tar_mamba", auxiliary_tasks=True):
    inputs = {
        "event_input": layers.Input(shape=(max_window,), dtype=tf.int32, name="event_input"),
        "type_input": layers.Input(shape=(max_window,), dtype=tf.int32, name="type_input"),
        "room_input": layers.Input(shape=(max_window,), dtype=tf.int32, name="room_input"),
        "state_input": layers.Input(shape=(max_window,), dtype=tf.int32, name="state_input"),
        "time_input": layers.Input(shape=(max_window, time_dim), dtype=tf.float32, name="time_input"),
        "context_input": layers.Input(shape=(context_dim,), dtype=tf.float32, name="context_input")
    }

    dim = cfg.get("model_dim", 64)

    e_event = layers.Embedding(num_events, dim)(inputs["event_input"])
    e_type = layers.Embedding(num_types, dim)(inputs["type_input"])
    e_room = layers.Embedding(num_rooms, dim)(inputs["room_input"])
    e_state = layers.Embedding(num_states, dim)(inputs["state_input"])

    x_discrete = layers.Concatenate(axis=-1)([e_event, e_type, e_room, e_state])
    x_discrete = layers.Dense(dim, activation="relu")(x_discrete)

    context = layers.RepeatVector(max_window)(inputs["context_input"])
    x_full = layers.Concatenate(axis=-1)([x_discrete, context, inputs["time_input"]])
    x_features = layers.Dense(dim, activation="swish")(x_full)

    x_features = layers.LayerNormalization()(x_features)
    x_features = layers.SpatialDropout1D(cfg.get("dropout", 0.2))(x_features)

    x_features = layers.Conv1D(filters=dim, kernel_size=4, padding="causal", groups=dim, activation="swish")(x_features)
    x_features = layers.LayerNormalization()(x_features)

    delta_t = inputs["time_input"][:, :, 4:5]
    rnn_input = layers.Concatenate(axis=-1)([x_features, delta_t])

    if backbone_type == "tar_mamba":
        x = TARMambaV1Backbone(units=dim, dropout=cfg.get("dropout", 0.2))(rnn_input)
    else:
        raise ValueError(f"Only 'tar_mamba' is supported in this ablation.")

    boundary_hidden = layers.Dense(dim // 2, activation="swish")(x)
    boundary_hidden = layers.Dropout(cfg.get("dropout", 0.2))(boundary_hidden)
    boundary_out = layers.Dense(1, activation="sigmoid", name="boundary_seq")(boundary_hidden)

    boundary_gate = layers.Conv1D(1, kernel_size=3, padding="same", activation="sigmoid", name="boundary_gate")(x)
    x_weighted = layers.Multiply()([x, 1.0 + boundary_gate])
    x_pooled = AttentionPooling()(x_weighted)

    # ============================================================
    # ACTIVITY CLASSIFICATION HEAD
    # ============================================================

    activity_hidden = layers.Dense(
        dim,
        activation="swish"
    )(x_pooled)

    activity_hidden = layers.LayerNormalization()(
        activity_hidden
    )

    activity_hidden = layers.Dropout(
        cfg.get("dropout", 0.2)
    )(activity_hidden)

    activity_hidden = layers.Dense(
        dim // 2,
        activation="swish"
    )(activity_hidden)

    activity_hidden = layers.Dropout(
        cfg.get("dropout", 0.2)
    )(activity_hidden)

    activity_out = layers.Dense(
        num_classes,
        activation="softmax",
        name="activity"
    )(activity_hidden)

    if auxiliary_tasks:
        return tf.keras.Model(inputs=inputs, outputs={"activity": activity_out, "boundary_seq": boundary_out})

    return tf.keras.Model(inputs=inputs, outputs=activity_out)