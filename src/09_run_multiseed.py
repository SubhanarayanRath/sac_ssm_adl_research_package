\
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for seed in [int(x) for x in args.seeds.split(",")]:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        temp = Path(f"config/seed_{seed}.yaml")
        temp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        name = f"sac_ssm_seed_{seed}"
        subprocess.run([
            sys.executable, "src/06_train_sac_ssm.py",
            "--config", str(temp), "--name", name
        ], check=True)
        subprocess.run([
            sys.executable, "src/07_evaluate.py",
            "--config", str(temp),
            "--model", f"models/{name}.keras",
            "--prefix", name
        ], check=True)


if __name__ == "__main__":
    main()
