# SAC-SSM for Ambient ADL Recognition

This repository implements a research-oriented pipeline for **irregular, event-driven smart-home Activity of Daily Living (ADL) recognition** using the WSU-CASAS Aruba dataset.

The current codebase focuses on the **SAC-SSM / TAR-Mamba v1 research pipeline**, with chronological splitting, transition-aware targets, irregular-event processing, and state-space sequence models.

## Proposed Research Direction

The repository contains implementations and experiments around:

- **SAC-SSM**: Semantic Adaptive-Context Selective State-Space Model
- Irregular event-driven sensor sequences
- Sensor and temporal representations
- Continuous-time / time-gap-aware sequence modeling
- Selective state-space recurrence
- Adaptive context modeling
- Transition-aware activity prediction
- TAR-Mamba v1
- GRU, selective Mamba, Momentum-Mamba and related backbones
- Evaluation and comparison of sequence models

The architecture is designed for smart-home activity recognition without requiring cameras, making it suitable for privacy-preserving ambient sensing.

---

## Dataset

The primary dataset is:

**WSU-CASAS Aruba**

The dataset should be placed under the local `data/` directory.

Raw and processed datasets are intentionally excluded from version control.

The repository is structured so that dataset-specific paths can be configured without committing the actual dataset.

---

## Current Pipeline

The current repository follows this general workflow:

```text
Raw Aruba event data
        |
        v
01_preprocess.py
        |
        v
01a_transition_targets.py
        |
        v
02_make_windows.py
        |
        v
13_make_transition_windows.py
        |
        v
Model training
        |
        v
22_train_tar_mamba_v1.py
        |
        v
23_evaluate_tar_mamba.py
```

The repository also contains reusable model components under:

```text
src/model/
├── __init__.py
├── layers.py
├── modular_sac.py
├── sac_ssm.py
└── backbones/
    ├── __init__.py
    ├── base.py
    ├── factory.py
    ├── irregular_gru.py
    ├── momentum_mamba.py
    ├── selective_mamba.py
    └── tar_mamba_v1.py
```

Additional data and loss utilities are located in:

```text
src/
├── tar_data_utils.py
└── losses.py
```

---

## Important Design Choice: Chronological Splitting

A major objective of the project is to avoid leakage caused by randomly splitting highly overlapping temporal windows.

The intended workflow is:

```text
Raw events
    ↓
Chronological train / validation / test split
    ↓
Window construction
    ↓
Model training
    ↓
Evaluation
```

rather than:

```text
Raw events
    ↓
Create overlapping windows
    ↓
Randomly split windows
    ↓
Train / test
```

Creating overlapping windows before splitting can result in nearly identical temporal samples appearing in both training and testing data.

The chronological approach is therefore treated as the primary experimental protocol.

---

## Current Source Structure

### Preprocessing

```text
src/01_preprocess.py
```

Preprocesses the raw sensor-event data into the representation required by the downstream pipeline.

### Transition Targets

```text
src/01a_transition_targets.py
```

Generates transition-aware targets used by the transition-oriented experiments.

### Standard Windows

```text
src/02_make_windows.py
```

Creates temporal windows from the appropriately split/preprocessed event stream.

### Transition Windows

```text
src/13_make_transition_windows.py
```

Creates windows specifically for transition-aware experiments.

### TAR-Mamba v1 Training

```text
src/22_train_tar_mamba_v1.py
```

Trains the current TAR-Mamba v1 model.

### TAR-Mamba v1 Evaluation

```text
src/23_evaluate_tar_mamba.py
```

Evaluates trained models and produces the relevant recognition metrics.

---

## Model Package

The model implementations have been reorganized under:

```text
src/model/
```

The package contains reusable components for different sequence backbones.

### Available Backbones

- Irregular GRU
- Selective Mamba
- Momentum Mamba
- TAR-Mamba v1

The backbone factory allows models to be selected without duplicating the surrounding training infrastructure.

---

## Running the Pipeline

Create the research environment:

```bash
conda create -n sac_ssm_adl python=3.10 -y
conda activate sac_ssm_adl
pip install -r requirements.txt
```

Then run the preprocessing stages according to the current experiment configuration.

For example:

```bash
python src/01_preprocess.py --config config/default.yaml
python src/01a_transition_targets.py --config config/default.yaml
python src/02_make_windows.py --config config/default.yaml
```

For transition experiments:

```bash
python src/13_make_transition_windows.py --config config/default.yaml
```

Train TAR-Mamba v1:

```bash
python src/22_train_tar_mamba_v1.py --config config/default.yaml
```

Evaluate:

```bash
python src/23_evaluate_tar_mamba.py --config config/default.yaml
```

> **Note:** The exact arguments and configuration options should be checked with `--help` for the current implementation.

---

## Dependencies

The model and training pipeline use Python scientific-computing and deep-learning dependencies specified in:

```text
requirements.txt
```

Some model implementations, including TAR-Mamba v1, require **TensorFlow**.

A missing TensorFlow installation will produce an error such as:

```text
ModuleNotFoundError: No module named 'tensorflow'
```

This is an environment/dependency issue, not a repository-structure issue.

The package itself can still be imported independently when the TensorFlow-dependent modules are not imported.

---

## Research Experiments

The research is focused on evaluating whether irregular-event and state-space modeling can improve ambient ADL recognition while maintaining a realistic temporal evaluation protocol.

Experiments may include comparisons between:

1. Conventional machine-learning baselines.
2. Recurrent sequence models.
3. Selective state-space models.
4. Momentum-based state-space models.
5. TAR-Mamba v1.
6. SAC-SSM variants.
7. Standard activity recognition.
8. Transition-aware activity recognition.
9. Different context and temporal representations.
10. Ablation studies of the proposed components.

The exact experiment set is maintained in the source code and configuration files rather than relying on obsolete scripts from earlier versions of the repository.

---

## Evaluation

The primary evaluation should consider metrics appropriate for imbalanced multi-class activity recognition, including:

- Accuracy
- Macro-F1
- Weighted-F1
- Balanced accuracy
- Matthews correlation coefficient (MCC)
- Cohen's kappa
- Per-class precision
- Per-class recall
- Per-class F1

Where applicable, model size, inference latency, and calibration can also be reported.

---

## Repository Hygiene

Large datasets, generated results, model checkpoints, logs, and other local artifacts are intentionally excluded from version control.

The `.gitignore` covers directories such as:

```text
data/
results/
result/
research/logs/
research/results/
research/models/
baselines/
```

The repository therefore contains the **reproducible source code and configuration**, while locally generated datasets and experiment artifacts remain outside Git.

---

## Project Status

The repository is under active research development.

The current codebase has been cleaned and reorganized around the newer model package and TAR-Mamba v1 workflow. Older experimental scripts that are no longer part of the active pipeline have been removed rather than being presented as supported commands.

The main objective is to establish a reproducible, leakage-resistant experimental framework for privacy-preserving ambient ADL recognition using irregular sensor-event sequences and state-space models.
