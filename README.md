# SAC-SSM for Ambient ADL Recognition

This package implements a publication-oriented pipeline for irregular, event-driven smart-home activity recognition.

## Proposed model

**SAC-SSM**: Semantic Adaptive-Context Selective State-Space Model.

The model combines:

- factorized event embeddings,
- sensor-type and optional room semantics,
- cyclic absolute time features,
- explicit time-gap information,
- short, medium, and long context branches,
- continuous-time selective state-space recurrence,
- learned adaptive fusion across context scales,
- self-supervised next-event and next-gap auxiliary tasks,
- class-balanced focal loss,
- chronological split-before-windowing.

The package is designed for WSU-CASAS Aruba first. It can later be extended to Milan, Cairo, and a custom smart-home dataset.

## Why this differs from the earlier code

The earlier code used heavily overlapping windows followed by random splitting. This can place nearly identical windows in training and testing. The present package assigns chronological train, validation, and test blocks first. Windows are created only after the split.

## Setup

```bash
conda create -n sac_ssm_adl python=3.10 -y
conda activate sac_ssm_adl
pip install -r requirements.txt
```

Place Aruba data at either:

```text
data/raw/aruba.txt
```

or provide a CSV file with recognizable timestamp, sensor, state, and activity columns.

## Execution

```bash
python src/01_preprocess.py --config config/default.yaml
python src/02_make_windows.py --config config/default.yaml
python src/03_train_ml_baselines.py --config config/default.yaml
python src/04_train_lstm_baseline.py --config config/default.yaml
python src/04b_train_previous_hybrid.py --config config/default.yaml
python src/05_pretrain_sac_ssm.py --config config/default.yaml
python src/06_train_sac_ssm.py --config config/default.yaml
python src/07_evaluate.py --config config/default.yaml
```

Ablations:

```bash
python src/08_run_ablations.py --config config/default.yaml
```

Repeated seeds:

```bash
python src/09_run_multiseed.py --config config/default.yaml
```

TFLite:

```bash
python src/10_convert_tflite.py --config config/default.yaml
```

Simulated real-time stream:

```bash
python src/11_realtime_simulation.py --config config/default.yaml
```

## Main paper experiments

1. Random Forest and Extra Trees.
2. BiLSTM baseline.
3. Earlier CNN-Transformer-BiLSTM-Attention baseline.
4. SAC-SSM without self-supervision.
5. Full SAC-SSM.
6. Single-scale vs multi-scale context.
7. Without continuous-time decay.
8. Without semantic embeddings.
9. Without auxiliary next-event and next-gap tasks.
10. Random split as a supplementary experiment only.
11. Chronological split as the primary result.

Primary metrics: macro-F1, balanced accuracy, weighted-F1, accuracy, MCC, Cohen's kappa, per-class F1, calibration error, model size, and inference latency.
