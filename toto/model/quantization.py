# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""Utility functions for post-training quantization."""

import torch
from torch import nn


def quantize_for_inference(model: nn.Module) -> nn.Module:
    """Apply dynamic post-training quantization for inference.

    This uses :func:`torch.quantization.quantize_dynamic` to convert supported
    layers (currently ``nn.Linear``) to int8 representations which can speed up
    CPU inference without requiring additional calibration data.

    Args:
        model: The trained model to quantize.

    Returns:
        A quantized copy of ``model`` suitable for CPU inference.
    """

    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
