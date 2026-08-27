\
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("full", []),
    ("no_pretrain", ["--no-pretrain"]),
    ("single_scale", ["--single-scale"]),
    ("no_semantics", ["--no-semantics"]),
    ("no_time_decay", ["--no-time-decay"]),
    ("no_aux", ["--no-aux"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    for name, flags in EXPERIMENTS:
        model_name = f"ablation_{name}"
        train_cmd = [
            sys.executable, "src/06_train_sac_ssm.py",
            "--config", args.config,
            "--name", model_name,
        ] + flags
        subprocess.run(train_cmd, check=True)

        eval_cmd = [
            sys.executable, "src/07_evaluate.py",
            "--config", args.config,
            "--model", f"models/{model_name}.keras",
            "--prefix", model_name,
        ]
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
