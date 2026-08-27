# Research gap and final novelty

## What should not be used as the main novelty

The main claim should not be:

- only a CNN-Transformer-BiLSTM-Attention architecture,
- only self-supervised next-event prediction,
- only activity-boundary detection,
- only floorplan trajectory modelling,
- only textual descriptions of sensor events,
- only GAN-based augmentation.

Each of these directions already has close recent work.

## Recommended primary contribution

**SAC-SSM: Semantic Adaptive-Context Selective State-Space Model for Ambient ADL Recognition**

### Core idea

The model treats an ambient stream as an irregular sequence. Memory should not decay at the same rate when two events are one second apart and when they are one hour apart. The proposed state-space cell therefore uses the observed time gap to control how much previous state is retained.

A single window size is also avoided. Three histories end at the same current event:

- short context: 20 events,
- medium context: 50 events,
- long context: 100 events.

Each history is encoded separately. A learned gate assigns a weight to each context scale. This allows a short activity to rely on recent evidence while a long routine can retain older context.

### Model inputs

Each event contains:

- sensor-state identity,
- sensor type,
- room semantics when available,
- sensor state,
- hour-of-day,
- day-of-week,
- logarithmic inter-event gap.

### Learning objectives

Stage 1: self-supervised pretraining

- predict the next sensor event,
- predict the next time-gap bin.

Stage 2: supervised fine-tuning

- classify the current activity,
- keep low-weight auxiliary next-event and next-gap losses.

### Evaluation protocol

- chronological split before windows,
- no train/test overlap from adjacent windows,
- macro-F1 as the primary metric,
- balanced accuracy,
- per-class F1,
- MCC,
- Cohen's kappa,
- calibration error,
- parameter count,
- mean and p95 latency.

## Expected claim

The publishable claim should be based on evidence, not architecture size:

> An adaptive continuous-time state-space representation improves minority-class recognition and temporal robustness in irregular ambient sensor streams while remaining suitable for edge deployment.

This claim must be supported by ablation results and repeated seeds.
