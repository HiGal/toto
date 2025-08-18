"""Triton Inference Server wrapper for Toto.

This module provides a minimal integration layer between the Toto model and the
NVIDIA Triton Inference Server using the Python backend.  The wrapper exposes a
``TritonPythonModel`` class that can be dropped into a Triton model repository
and used to serve Toto forecasts.  Only the pieces required for loading the
model and producing forecasts are implemented; advanced features such as
batching or dynamic batching can be added later if needed.

Example Triton model repository structure::

    models/
      toto/
        1/
          model.safetensors
          model.py  # <- this file
        config.pbtxt

The config should declare the inputs listed in ``execute`` below.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ``triton_python_backend_utils`` is only available inside the Triton Python
# backend runtime.  Import lazily so that unit tests or other environments that
# do not provide the module can still import this file without failing.
try:  # pragma: no cover - small utility guard
    import triton_python_backend_utils as pb_utils
except Exception:  # pragma: no cover - the dependency is optional
    pb_utils = None  # type: ignore

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


class TritonPythonModel:  # pragma: no cover - exercised in Triton runtime
    """Triton entry point for serving Toto forecasts.

    The class follows the interface required by the Triton Python backend.  It
    expects the model checkpoint (``model.safetensors`` by default) to live
    under the version directory of the Triton model repository.  During
    ``initialize`` the checkpoint is loaded and wrapped with :class:`TotoForecaster`.

    Requests handled by ``execute`` must provide the following tensors:

    - ``series``: ``float32``/``float64`` of shape ``(batch, variates, time)``.
    - ``padding_mask``: ``bool`` mask of same shape as ``series`` indicating
      valid values.
    - ``id_mask``: ``int32``/``int64`` mask of same shape.
    - ``timestamp_seconds``: ``int32``/``int64`` timestamp per element.
    - ``time_interval_seconds``: ``int32``/``int64`` of shape ``(batch, variates)``.
    - ``prediction_length``: scalar ``int32`` specifying the horizon.
    - ``num_samples`` *(optional)*: scalar ``int32`` for stochastic sampling.

    The response contains ``mean`` and, when sampling, ``samples`` tensors.
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        """Load the Toto checkpoint and create the forecaster."""

        if pb_utils is None:  # pragma: no cover - safety for non-triton env
            raise RuntimeError("triton_python_backend_utils is required inside Triton runtime")

        model_config = json.loads(args["model_config"])
        repo_path = args["model_repository"]
        version = args["model_version"]

        # Allow overriding checkpoint name via config parameters.
        params = model_config.get("parameters", {})
        checkpoint = params.get("checkpoint", {}).get("string_value", "model.safetensors")
        compile_flag = params.get("compile", {}).get("string_value", "false").lower() in {"1", "true", "yes", "on"}
        quantize_flag = params.get("quantize", {}).get("string_value", "false").lower() in {"1", "true", "yes", "on"}
        num_threads_val = params.get("num_threads", {}).get("string_value")
        num_threads = int(num_threads_val) if num_threads_val is not None else None

        checkpoint_path = os.path.join(repo_path, version, checkpoint)

        toto = Toto.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.forecaster = TotoForecaster(
            toto.model,
            compile=compile_flag,
            num_threads=num_threads,
            quantize=quantize_flag,
        )

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Run Toto forecasts for a list of Triton requests."""

        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            series = self._torch_tensor(request, "series")
            padding_mask = self._torch_tensor(request, "padding_mask", dtype=torch.bool)
            id_mask = self._torch_tensor(request, "id_mask", dtype=torch.int32)
            timestamp_seconds = self._torch_tensor(request, "timestamp_seconds", dtype=torch.int32)
            time_interval_seconds = self._torch_tensor(request, "time_interval_seconds", dtype=torch.int32)

            prediction_length = int(pb_utils.get_input_tensor_by_name(request, "prediction_length").as_numpy().item())
            num_samples_tensor = pb_utils.get_input_tensor_by_name(request, "num_samples")
            num_samples: Optional[int] = None
            if num_samples_tensor is not None:
                num_samples = int(num_samples_tensor.as_numpy().item())

            inputs = MaskedTimeseries(
                series=series,
                padding_mask=padding_mask,
                id_mask=id_mask,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
            )

            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

            output_tensors = [
                pb_utils.Tensor("mean", forecast.mean.cpu().numpy()),
            ]
            if forecast.samples is not None:
                output_tensors.append(pb_utils.Tensor("samples", forecast.samples.cpu().numpy()))

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def _torch_tensor(
        self,
        request: pb_utils.InferenceRequest,
        name: str,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Utility to fetch a tensor from the request as a Torch tensor."""

        tensor = pb_utils.get_input_tensor_by_name(request, name)
        array = tensor.as_numpy()
        torch_tensor = torch.from_numpy(array)
        if dtype is not None:
            torch_tensor = torch_tensor.to(dtype)
        return torch_tensor
