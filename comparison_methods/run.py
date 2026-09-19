#!/usr/bin/env python3
"""Verify the IresFM comparison methods with synthetic inputs."""

from __future__ import annotations

import argparse
import gc
import random

import numpy as np
import torch

from .configs import METHOD_SPECS
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-test", help="check all documented model settings")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--selected-only", action="store_true")
    return parser.parse_args()


def smoke_test(device_name: str, selected_only: bool) -> None:
    device = torch.device(device_name)
    completed = 0
    for name, spec in METHOD_SPECS.items():
        configs = (spec.selected,) if selected_only else (1, 2, 3)
        weather_dim = 15 if spec.technology == "wind" else 12
        for config_index in configs:
            set_seed(42)
            model = build_model(name, config_index).to(device)
            weather_past = torch.randn(1, 120, weather_dim, device=device)
            weather_future = torch.randn(1, 120, weather_dim, device=device)
            power_past = torch.randn(1, 480, device=device)
            output = model(weather_past, weather_future, power_past)
            if tuple(output.shape) != (1, 480) or not torch.isfinite(output).all():
                raise AssertionError(f"{name} config {config_index}: invalid output")
            output.square().mean().backward()
            if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
                raise AssertionError(f"{name} config {config_index}: no gradients")
            completed += 1
            print(f"PASS {name} config={config_index} output=(1, 480)")
            del model, weather_past, weather_future, power_past, output
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"Completed {completed} forward/backward checks.")


def main() -> None:
    args = parse_args()
    smoke_test(args.device, args.selected_only)


if __name__ == "__main__":
    main()
