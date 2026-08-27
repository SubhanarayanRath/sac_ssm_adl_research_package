\
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config_utils import load_config
from losses import ClassBalancedFocalLoss
from models.layers import ContinuousTimeSSMCell, ContinuousTimeBiSSM, AttentionPooling, WeightedScaleFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/sac_ssm_full.keras")
    parser.add_argument("--output", default="models/sac_ssm_full.tflite")
    args = parser.parse_args()

    _ = load_config(args.config)
    custom = {
        "ClassBalancedFocalLoss": ClassBalancedFocalLoss,
        "ContinuousTimeSSMCell": ContinuousTimeSSMCell,
        "ContinuousTimeBiSSM": ContinuousTimeBiSSM,
        "AttentionPooling": AttentionPooling,
        "WeightedScaleFusion": WeightedScaleFusion,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom, compile=False)

    # Export only the activity output for deployment.
    deploy_model = tf.keras.Model(model.inputs, model.outputs[0], name="SAC_SSM_Activity")
    converter = tf.lite.TFLiteConverter.from_keras_model(deploy_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite = converter.convert()
    Path(args.output).write_bytes(tflite)
    print(f"Saved {args.output} ({len(tflite) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
