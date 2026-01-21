i#!/usr/bin/env python3
"""
Local training script - runs training, quantization, and benchmarking on the local machine.
This is the local equivalent of the Modal-based training script.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import argparse
import glob
from typing import Optional, List

import train


def training_run(run_name: str):
    """Run the training locally."""
    return train.do_training_run(run_name=run_name)


def quantization_run(fp32_model_path: str):
    """Run quantization locally."""
    return train.do_quantization_run(fp32_model_path=fp32_model_path)


def benchmark_run(model_root: str):
    """Run benchmarking locally."""
    model_paths = glob.glob(f"{model_root}/*.onnx")
    return train.do_benchmark_run(model_paths=model_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Local training, quantization, and benchmarking script"
    )
    parser.add_argument(
        "--training-run-name",
        type=str,
        default=None,
        help="Name for the training run",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        help="Path to FP32 model to quantize",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to model root directory containing ONNX files to benchmark",
    )

    args = parser.parse_args()

    if args.training_run_name is not None:
        print(f"Starting training run: {args.training_run_name}")
        training_run(run_name=args.training_run_name)

    if args.quantize is not None:
        print(f"Starting quantization for: {args.quantize}")
        quantization_run(fp32_model_path=args.quantize)

    if args.benchmark is not None:
        print(f"Starting benchmark for models in: {args.benchmark}")
        benchmark_run(model_root=args.benchmark)

    if all(x is None for x in [args.training_run_name, args.quantize, args.benchmark]):
        parser.print_help()


if __name__ == "__main__":
    main()
