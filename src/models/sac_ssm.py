from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers import ContinuousTimeBiSSM, AttentionPooling, WeightedScaleFusion

# Import our new swappable backbones
from .backbones.irregular_gru import IrregularTimeGRU

def slice_last(x, length: int):
    return x[:, -length:]

def build_sac_ssm(
    num_events: int,
    num_types: int,
    num_rooms: int,
    num_states: int,
    num_classes: int,
    num_gap_bins: int,
    max_window: int,
    time_dim: int,
    context_dim: int,
    cfg: dict,
    single_scale: bool = False,
    no_semantics: bool = False,
    no_time_decay: bool = False,
    backbone_type: str = "ct_bissm",  # <-- NEW: Allows swapping the core
):
    mc = cfg["model"]
    dc = cfg["data"]

    # 1. Inputs
    event_in = layers.Input((max_window,), dtype="int32", name="event_input")
    type_in = layers.Input((max_window,), dtype="int32", name="type_input")
    room_in = layers.Input((max_window,), dtype="int32", name="room_input")
    state_in = layers.Input((max_window,), dtype="int32", name="state_input")
    time_in = layers.Input((max_window, time_dim), dtype="float32", name="time_input")
    context_in = layers.Input((context_dim,), dtype="float32", name="context_input")

    # 2. Embeddings
    event_emb = layers.Embedding(num_events, int(mc["event_embedding_dim"]), name="event_embedding")
    type_emb = layers.Embedding(num_types, int(mc["type_embedding_dim"]), name="type_embedding")
    room_emb = layers.Embedding(num_rooms, int(mc["room_embedding_dim"]), name="room_embedding")
    state_emb = layers.Embedding(num_states, int(mc["state_embedding_dim"]), name="state_embedding")
    time_proj = layers.Dense(int(mc["time_projection_dim"]), activation="tanh", name="time_projection")

    def encode_scale(length: int, name: str):
        ev = layers.Lambda(lambda z: z[:, -length:], name=f"{name}_slice_event")(event_in)
        ty = layers.Lambda(lambda z: z[:, -length:], name=f"{name}_slice_type")(type_in)
        ro = layers.Lambda(lambda z: z[:, -length:], name=f"{name}_slice_room")(room_in)
        st = layers.Lambda(lambda z: z[:, -length:], name=f"{name}_slice_state")(state_in)
        ti = layers.Lambda(lambda z: z[:, -length:, :], name=f"{name}_slice_time")(time_in)

        parts = [event_emb(ev), state_emb(st), time_proj(ti)]
        if not no_semantics:
            parts.extend([type_emb(ty), room_emb(ro)])

        x = layers.Concatenate(name=f"{name}_concat")(parts)
        x = layers.Dense(int(mc["model_dim"]), activation="relu", name=f"{name}_input_projection")(x)
        x = layers.SeparableConv1D(
            int(mc["model_dim"]), kernel_size=3, padding="same", activation="relu", name=f"{name}_local_conv"
        )(x)
        x = layers.Dropout(float(mc["dropout"]))(x)

        delta = layers.Lambda(lambda z: z[:, :, -1:], name=f"{name}_delta")(ti)
        ssm_input = layers.Concatenate(name=f"{name}_ssm_input")([x, delta])

        # --- THE SWAPPABLE BACKBONE ---
        if backbone_type == "ct_bissm":
            x = ContinuousTimeBiSSM(
                int(mc["ssm_hidden_dim"]),
                use_time_decay=not no_time_decay,
                dropout=float(mc["dropout"]),
                name=f"{name}_ct_bissm",
            )(ssm_input)
        elif backbone_type == "gruwe":
            x = IrregularTimeGRU(
                int(mc["ssm_hidden_dim"]),
                dropout=float(mc["dropout"]),
                name=f"{name}_gruwe",
            )(ssm_input)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
        # ------------------------------

        x = AttentionPooling(name=f"{name}_attention_pool")(x)
        x = layers.Dense(int(mc["branch_output_dim"]), activation="relu", name=f"{name}_branch_dense")(x)
        return x

    # 3. Multi-scale fusion
    if single_scale:
        lengths = [int(dc["long_window"])]
    else:
        lengths = [int(dc["short_window"]), int(dc["medium_window"]), int(dc["long_window"])]

    branch_vectors = [encode_scale(length, f"scale_{length}") for length in lengths]

    if len(branch_vectors) == 1:
        fused = branch_vectors[0]
        scale_weights = layers.Lambda(
            lambda z: tf.ones((tf.shape(z)[0], 1), dtype=z.dtype), name="scale_weights"
        )(fused)
    else:
        fused, scale_weights = WeightedScaleFusion(
            len(branch_vectors), hidden_dim=32, name="adaptive_scale_fusion"
        )([branch_vectors, context_in])

    x = layers.Concatenate(name="fused_with_context")([fused, context_in])
    x = layers.Dense(160, activation="relu", name="fusion_dense_1")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.35)(x)
    representation = layers.Dense(96, activation="relu", name="representation")(x)

    # 4. Outputs
    activity_out = layers.Dense(num_classes, activation="softmax", name="activity")(representation)
    next_event_out = layers.Dense(num_events, activation="softmax", name="next_event")(representation)
    gap_out = layers.Dense(num_gap_bins, activation="softmax", name="gap_bin")(representation)

    model = Model(
        inputs=[event_in, type_in, room_in, state_in, time_in, context_in],
        outputs=[activity_out, next_event_out, gap_out, scale_weights],
        name=f"SAC_ADL_{backbone_type.upper()}",
    )
    return model