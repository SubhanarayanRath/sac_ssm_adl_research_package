# Experiment matrix

## Main comparison

| Model | Split | Primary purpose |
|---|---|---|
| Random Forest | chronological | classical baseline |
| Extra Trees | chronological | strong tree baseline |
| BiLSTM | chronological | recurrent baseline |
| Earlier CNN-Transformer-BiLSTM-Attention | chronological | direct comparison with previous implementation |
| SAC-SSM without pretraining | chronological | architecture effect |
| Full SAC-SSM | chronological | final model |

## Ablation study

| Variant | Removed component |
|---|---|
| Full | none |
| No pretraining | self-supervised stage |
| Single scale | adaptive context |
| No semantics | sensor type and room embeddings |
| No time decay | continuous-time memory |
| No auxiliary tasks | next-event and next-gap losses |

## Statistical validation

Run five seeds. Report mean and standard deviation.

Recommended tests:

- bootstrap 95% confidence interval for macro-F1,
- Wilcoxon signed-rank test on day-wise macro-F1,
- McNemar test for paired prediction errors,
- Holm correction for multiple comparisons.

## Required paper figures

1. Full preprocessing and evaluation pipeline.
2. SAC-SSM architecture.
3. Class distribution.
4. Main confusion matrix.
5. Per-class F1 comparison.
6. Scale-weight analysis for short and long activities.
7. Reliability diagram.
8. Latency and model-size comparison.
