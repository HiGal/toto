#!/usr/bin/env python3
"""Launch Triton Inference Server for Toto with CPU optimizations."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


def prepare_repository(
    checkpoint: str,
    repo: Path,
    *,
    threads: int,
    compile_model: bool,
    quantize: bool,
) -> None:
    """Create a Triton model repository for Toto."""
    model_dir = repo / "toto" / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy model checkpoint and wrapper.
    shutil.copy(checkpoint, model_dir / "model.safetensors")
    wrapper_src = Path(__file__).with_name("triton_wrapper.py")
    shutil.copy(wrapper_src, model_dir / "model.py")

    config = f"""
    name: \"toto\"
    backend: \"python\"
    max_batch_size: 0
    input [
      {{ name: \"series\", data_type: TYPE_FP32, dims: [ -1, -1, -1 ] }},
      {{ name: \"padding_mask\", data_type: TYPE_BOOL, dims: [ -1, -1, -1 ] }},
      {{ name: \"id_mask\", data_type: TYPE_INT32, dims: [ -1, -1, -1 ] }},
      {{ name: \"timestamp_seconds\", data_type: TYPE_INT32, dims: [ -1, -1, -1 ] }},
      {{ name: \"time_interval_seconds\", data_type: TYPE_INT32, dims: [ -1, -1 ] }},
      {{ name: \"prediction_length\", data_type: TYPE_INT32, dims: [1] }},
      {{ name: \"num_samples\", data_type: TYPE_INT32, dims: [1], optional: true }}
    ]
    output [
      {{ name: \"mean\", data_type: TYPE_FP32, dims: [ -1, -1, -1 ] }},
      {{ name: \"samples\", data_type: TYPE_FP32, dims: [ -1, -1, -1, -1 ], optional: true }}
    ]
    parameters {{
      key: \"num_threads\"
      value {{ string_value: \"{threads}\" }}
    }}
    parameters {{
      key: \"compile\"
      value {{ string_value: \"{str(compile_model).lower()}\" }}
    }}
    parameters {{
      key: \"quantize\"
      value {{ string_value: \"{str(quantize).lower()}\" }}
    }}
    """
    (repo / "toto" / "config.pbtxt").write_text(textwrap.dedent(config))


def launch_server(repo: Path, *, threads: int) -> None:
    """Start Triton Inference Server."""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(threads))
    cmd = ["tritonserver", "--model-repository", str(repo)]
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to Toto checkpoint")
    parser.add_argument("--repo", default="model_repo", help="Model repository directory")
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU threads for PyTorch",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Compile model with torch.compile",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only create repository without starting server",
    )
    args = parser.parse_args()

    repo = Path(args.repo)
    prepare_repository(
        args.checkpoint,
        repo,
        threads=args.threads,
        compile_model=args.compile_model,
        quantize=args.quantize,
    )

    if not args.no_launch:
        launch_server(repo, threads=args.threads)


if __name__ == "__main__":
    main()
