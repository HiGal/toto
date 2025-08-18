# Triton CPU Inference

This directory contains a helper script to serve a Toto model using the
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
with optimizations for CPU execution.

## Preparing and launching the server

1. **Install Triton** – ensure the `tritonserver` binary is available and the
   Python backend is enabled.
2. **Run the helper script** to create a Triton model repository and launch the
   server:

   ```bash
   python toto/inference/run_triton_server.py \
       --checkpoint /path/to/model.safetensors \
       --repo model_repo \
       --threads 8 \
       --compile \
       --quantize
   ```

   - `--threads` sets the number of CPU threads used by PyTorch.
   - `--compile` compiles the model with `torch.compile`.
   - `--quantize` applies dynamic post-training quantization.
   - Use `--no-launch` to only create the repository without starting the
     server.

   The script copies the checkpoint and `triton_wrapper.py` into the model
   repository and starts `tritonserver` with `OMP_NUM_THREADS` set for the
   requested thread count.

3. **Send requests** using one of the Triton clients. For example, using the
   HTTP client:

   ```python
   import numpy as np
   import tritonclient.http as httpclient

   client = httpclient.InferenceServerClient("localhost:8000")
   # Prepare numpy arrays for the required tensors...
   result = client.infer("toto", inputs)
   print(result.as_numpy("mean"))
   ```

## Customizing the repository

`run_triton_server.py` writes a `config.pbtxt` that exposes parameters for
`num_threads`, `compile` and `quantize`. These values can be manually edited in
`model_repo/toto/config.pbtxt` for further tuning.
