# CPU Inference Optimization Problems

The following issues were identified when reviewing the CPU inference pipeline and have now been resolved:

1. **Use of `torch.no_grad` instead of `torch.inference_mode`.**
   - Several inference and scaling routines relied on `torch.no_grad`, which still incurs autograd setup costs. Replacing it with `torch.inference_mode` gives better performance on CPU.
   - *Fix applied*: switched relevant contexts and decorators to `torch.inference_mode` in inference and scaler modules.

2. **Repeated tensor concatenations in autoregressive loops.**
   - Functions like `generate_mean` and `generate_samples` repeatedly called `torch.cat` inside loops, causing frequent memory reallocations.
   - *Fix applied*: both routines now preallocate tensors to their final sizes and write predictions in place, eliminating repeated concatenations.

3. **Inefficient timestamp and array construction.**
   - `_generate` previously built timestamp arrays via Python loops and NumPy conversions, adding CPU overhead.
   - *Fix applied*: timestamps are now constructed directly with vectorized PyTorch operations.

4. **Potential gains from compilation and threading controls.**
   - Compiling the model with `torch.compile`/`torch.jit` or tuning `torch.set_num_threads` can yield additional CPU speedups depending on the environment.
   - *Fix applied*: `TotoForecaster` optionally compiles the model and allows setting the number of threads for CPU inference.

5. **Missing post-training quantization.**
   - Quantization can further accelerate CPU inference by using int8 weights.
   - *Fix applied*: added utilities to apply dynamic post-training quantization and integrated optional quantization into the forecaster.

