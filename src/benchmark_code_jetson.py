#!/usr/bin/env python3
"""
Jetson-oriented inference benchmark for this repository's baseline models.

The benchmark uses the same model factory as train_difflogic.py:

    baseline_models.make_baseline_model(model_kind, input_shape, ...)

By default it uses synthetic random samples with the common 15-channel x 5-band
feature shape, benchmarks DiffLogic-medium plus the MLP, 1-Conv, and Transformer
baselines, and runs ONNX Runtime on CPU for non-DiffLogic models when ONNX export
is available.
"""

import argparse
import contextlib
import csv
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baseline_models import count_trainable_parameters, make_baseline_model


DIFFLOGIC_MODEL_KIND = "difflogic_medium"
BASELINE_MODEL_KINDS = ["mlp_250k", "conv1d_250k", "transformer_250k"]
MODEL_KINDS = [DIFFLOGIC_MODEL_KIND] + BASELINE_MODEL_KINDS
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs/benchmarks"
DEFAULT_INPUT_SHAPE = (15, 5)


def parse_input_shape(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def infer_input_shape(feature_kind, hfd_name, exclude_channels):
    from train_helpers import load_alz_c_vs_a_dataset

    subjects, _ = load_alz_c_vs_a_dataset(
        feature_kind=feature_kind,
        hfd_name=hfd_name,
        exclude_channels=exclude_channels,
    )

    if not subjects:
        raise ValueError(f"No ALZ_FTD subjects found for feature_kind={feature_kind}")

    return tuple(int(value) for value in subjects[0]["x"].shape[1:])


def get_input_shape(args, exclude_channels):
    if args.input_shape is not None:
        return parse_input_shape(args.input_shape), "argument"

    if not args.infer_input_shape:
        return DEFAULT_INPUT_SHAPE, "default"

    try:
        return infer_input_shape(args.feature_kind, args.hfd_name, exclude_channels), "dataset"
    except Exception as exc:
        print(
            "Could not infer input shape from local datasets; "
            f"using default {DEFAULT_INPUT_SHAPE}. Reason: {exc}"
        )
        return DEFAULT_INPUT_SHAPE, "default"


def parse_exclude_channels(value):
    if value == "":
        return []

    return [channel.strip() for channel in value.split(",") if channel.strip()]


def set_thread_limits(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def make_random_input(batch_size, input_shape, device):
    shape = (batch_size,) + tuple(input_shape)
    return torch.rand(shape, dtype=torch.float32, device=device)


def make_random_difflogic_input(batch_size, input_shape, thermometer_bins, device):
    input_dim = int(np.prod(input_shape)) * thermometer_bins
    return torch.randint(0, 2, (batch_size, input_dim), dtype=torch.float32, device=device)


def is_difflogic_model(model_kind):
    return model_kind == DIFFLOGIC_MODEL_KIND


def benchmark_torch(model, x, warmup, iterations, device):
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(iterations):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start_time) / iterations * 1000.0


def get_file_size_kb(path):
    return path.stat().st_size / 1024.0 if path.exists() else 0.0


def save_state_dict_size_kb(model, path):
    torch.save(model.state_dict(), path)
    return get_file_size_kb(path)


def load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime is not installed"

    try:
        import onnx  # noqa: F401
    except ImportError:
        return None, "onnx is not installed"

    return ort, "available"


def make_onnx_session_options(ort, num_threads):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = num_threads
    return session_options


def export_onnx(model, example_input, output_path, opset_versions):
    errors = []

    for opset_version in opset_versions:
        try:
            torch.onnx.export(
                model,
                example_input,
                output_path,
                input_names=["x"],
                output_names=["logits"],
                dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=opset_version,
            )
            return opset_version
        except Exception as exc:
            errors.append(f"opset {opset_version}: {exc}")

    raise RuntimeError("; ".join(errors))


class TransformerOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.model.input_projection(x)
        class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)

        layer = self.model.encoder.layers[0]
        x = self.forward_encoder_layer(layer, x)
        return self.model.head(x[:, 0])

    def forward_encoder_layer(self, layer, x):
        attention_output = self.forward_self_attention(layer.self_attn, x)
        x = layer.norm1(x + attention_output)
        feedforward_output = layer.linear2(F.relu(layer.linear1(x)))
        return layer.norm2(x + feedforward_output)

    def forward_self_attention(self, attention, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        embed_dim = x.shape[2]
        num_heads = attention.num_heads
        head_dim = embed_dim // num_heads

        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.reshape_heads(query, batch_size, sequence_length, num_heads, head_dim)
        key = self.reshape_heads(key, batch_size, sequence_length, num_heads, head_dim)
        value = self.reshape_heads(value, batch_size, sequence_length, num_heads, head_dim)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, embed_dim)
        return attention.out_proj(context)

    def reshape_heads(self, x, batch_size, sequence_length, num_heads, head_dim):
        return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def make_onnx_export_model(model_kind, model):
    if model_kind == "transformer_250k":
        return TransformerOnnxWrapper(model)

    return model


def benchmark_onnx(session, x_numpy, warmup, iterations):
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: x_numpy})

    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: x_numpy})

    return (time.perf_counter() - start_time) / iterations * 1000.0


def benchmark_callable(fn, x, warmup, iterations):
    for _ in range(warmup):
        fn(x)

    start_time = time.perf_counter()
    for _ in range(iterations):
        fn(x)

    return (time.perf_counter() - start_time) / iterations * 1000.0


def run_quietly(fn, *args, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return fn(*args, **kwargs)


def compile_difflogic_model(model, output_path):
    import difflogic

    compiled_model = difflogic.CompiledLogicNet(
        model,
        num_bits=8,
        cpu_compiler="gcc",
        verbose=False,
    )
    run_quietly(compiled_model.compile, save_lib_path=str(output_path), opt_level=3, verbose=False)
    return compiled_model


def count_logic_gates(model):
    return sum(
        getattr(module, "out_dim", 0)
        for module in model.modules()
        if module.__class__.__name__ == "LogicLayer"
    )


def benchmark_model_kind(args, model_kind, input_shape, device, ort=None):
    model_device = str(device)
    try:
        make_model = run_quietly if is_difflogic_model(model_kind) else lambda fn, *a, **kw: fn(*a, **kw)
        model = make_model(
            make_baseline_model,
            model_kind,
            input_shape=input_shape,
            dropout=args.dropout,
            target_parameters=args.target_parameters,
            thermometer_bins=args.thermometer_bins,
            tau=args.tau,
            device=model_device,
        ).to(device)
    except ImportError as exc:
        return [
            {
                "model_kind": model_kind,
                "backend": f"torch_{device.type}_failed",
                "params": 0,
                "gates": 0,
                "size_kb": 0.0,
                "latency_ms": np.nan,
                "error": str(exc),
            }
        ]

    if is_difflogic_model(model_kind):
        x = make_random_difflogic_input(args.batch_size, input_shape, args.thermometer_bins, device)
        gates = count_logic_gates(model)
    else:
        x = make_random_input(args.batch_size, input_shape, device)
        gates = count_trainable_parameters(model)

    params = count_trainable_parameters(model)
    torch_artifact_path = args.output_dir / f"{model_kind}_{args.feature_kind}_torch.pt"
    try:
        torch_size_kb = save_state_dict_size_kb(model, torch_artifact_path)
        torch_artifact = torch_artifact_path.name
    except Exception:
        torch_size_kb = 0.0
        torch_artifact = ""

    results = []
    try:
        latency_ms = benchmark_torch(model, x, args.warmup, args.iterations, device)
        results.append(
            {
                "model_kind": model_kind,
                "backend": f"torch_{device.type}",
                "params": params,
                "gates": gates,
                "size_kb": torch_size_kb,
                "latency_ms": latency_ms,
                "artifact": torch_artifact,
            }
        )
    except Exception as exc:
        results.append(
            {
                "model_kind": model_kind,
                "backend": f"torch_{device.type}_failed",
                "params": params,
                "gates": gates,
                "size_kb": torch_size_kb,
                "latency_ms": np.nan,
                "artifact": torch_artifact,
                "error": str(exc),
            }
        )

    if is_difflogic_model(model_kind) and device.type == "cpu" and not args.skip_compiled_difflogic:
        so_path = args.output_dir / f"{model_kind}_{args.feature_kind}_compiled.so"
        try:
            compiled_model = compile_difflogic_model(model, so_path)
            x_bool = x.cpu().numpy().astype(bool)
            compiled_latency_ms = benchmark_callable(
                compiled_model,
                x_bool,
                args.warmup,
                args.iterations,
            )
            results.append(
                {
                    "model_kind": model_kind,
                    "backend": "difflogic_c_cpu",
                    "params": params,
                    "gates": gates,
                    "size_kb": get_file_size_kb(so_path),
                    "latency_ms": compiled_latency_ms,
                    "artifact": so_path.name,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_kind": model_kind,
                    "backend": "difflogic_c_cpu_failed",
                    "params": params,
                    "gates": gates,
                    "size_kb": 0.0,
                    "latency_ms": np.nan,
                    "artifact": so_path.name,
                    "error": str(exc),
                }
            )

    if is_difflogic_model(model_kind) or device.type != "cpu" or ort is None or args.skip_onnx:
        return results

    onnx_path = args.output_dir / f"{model_kind}_{args.feature_kind}.onnx"
    try:
        export_model = make_onnx_export_model(model_kind, model.cpu().eval())
        opset_version = export_onnx(export_model, x.cpu(), onnx_path, args.onnx_opsets)
        session_options = make_onnx_session_options(ort, args.num_threads)
        session = ort.InferenceSession(
            str(onnx_path),
            session_options,
            providers=["CPUExecutionProvider"],
        )
        onnx_latency_ms = benchmark_onnx(
            session,
            x.cpu().numpy(),
            args.warmup,
            args.iterations,
        )
        results.append(
            {
                "model_kind": model_kind,
                "backend": "onnx_cpu",
                "params": params,
                "gates": gates,
                "size_kb": get_file_size_kb(onnx_path),
                "latency_ms": onnx_latency_ms,
                "onnx_opset": opset_version,
                "artifact": onnx_path.name,
            }
        )
    except Exception as exc:
        results.append(
            {
                "model_kind": model_kind,
                "backend": "onnx_cpu_failed",
                "params": params,
                "gates": gates,
                "size_kb": 0.0,
                "latency_ms": np.nan,
                "onnx_opset": "",
                "artifact": onnx_path.name,
                "error": str(exc),
            }
        )

    return results


def print_results(results, batch_size, input_shape):
    print("\n" + "=" * 94)
    print("Baseline Inference Benchmark".center(94))
    print("=" * 94)
    print(f"Input shape: {input_shape} | Batch: {batch_size}")
    print("-" * 94)
    print(
        f"{'Model':<20} {'Backend':<18} {'Params':>12} {'Gates':>10} {'Size(KB)':>10} "
        f"{'Latency':>14} {'Throughput':>14}"
    )
    print("-" * 94)

    for result in results:
        latency_ms = result["latency_ms"]
        throughput = batch_size / latency_ms * 1000.0 if np.isfinite(latency_ms) else np.nan
        size_text = f"{result['size_kb']:.1f}" if result["size_kb"] > 0 else "-"
        latency_text = f"{latency_ms:.4f} ms" if np.isfinite(latency_ms) else "failed"
        throughput_text = f"{throughput:,.0f}/s" if np.isfinite(throughput) else "-"
        print(
            f"{result['model_kind']:<20} {result['backend']:<18} "
            f"{result['params']:>12,} {result.get('gates', 0):>10,} {size_text:>10} "
            f"{latency_text:>14} {throughput_text:>14}"
        )

        if "error" in result:
            print(f"  Error: {result['error']}")

    print("=" * 94)


def save_results(results, args, input_shape):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / args.output_csv
    fieldnames = [
        "feature_kind",
        "hfd_name",
        "input_shape",
        "batch_size",
        "iterations",
        "warmup",
        "model_kind",
        "backend",
        "params",
        "gates",
        "size_kb",
        "artifact",
        "latency_ms",
        "throughput_per_second",
        "onnx_opset",
        "error",
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            latency_ms = result["latency_ms"]
            throughput = args.batch_size / latency_ms * 1000.0 if np.isfinite(latency_ms) else np.nan
            writer.writerow(
                {
                    "feature_kind": args.feature_kind,
                    "hfd_name": args.hfd_name if args.feature_kind == "hfd" else "",
                    "input_shape": "x".join(str(value) for value in input_shape),
                    "batch_size": args.batch_size,
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "model_kind": result["model_kind"],
                    "backend": result["backend"],
                    "params": result["params"],
                    "gates": result.get("gates", 0),
                    "size_kb": "{:.3f}".format(result["size_kb"]),
                    "artifact": result.get("artifact", ""),
                    "latency_ms": "{:.6f}".format(latency_ms) if np.isfinite(latency_ms) else "",
                    "throughput_per_second": "{:.3f}".format(throughput) if np.isfinite(throughput) else "",
                    "onnx_opset": result.get("onnx_opset", ""),
                    "error": result.get("error", ""),
                }
            )

    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", default="all", choices=["all"] + MODEL_KINDS)
    parser.add_argument("--feature-kind", default="psd", choices=["psd", "hfd"])
    parser.add_argument("--hfd-name", default="kmax_16")
    parser.add_argument(
        "--input-shape",
        default=None,
        help="Comma-separated shape, e.g. 15,5. Defaults to the common 15 channels x 5 bands.",
    )
    parser.add_argument(
        "--infer-input-shape",
        action="store_true",
        help="Try to infer input shape from local ALZ_FTD feature files.",
    )
    parser.add_argument("--exclude-channels", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--target-parameters", type=int, default=250_000)
    parser.add_argument("--thermometer-bins", type=int, default=15)
    parser.add_argument("--tau", type=float, default=30.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-compiled-difflogic", action="store_true")
    parser.add_argument(
        "--onnx-opsets",
        default="16,15,14,13",
        help="Comma-separated ONNX opsets to try in order.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", default="baseline_jetson_benchmark.csv")
    args = parser.parse_args()
    args.onnx_opsets = parse_opset_list(args.onnx_opsets)
    return args


def parse_opset_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    args = parse_args()
    set_thread_limits(args.num_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    exclude_channels = parse_exclude_channels(args.exclude_channels)
    input_shape, input_shape_source = get_input_shape(args, exclude_channels)

    device = torch.device(args.device)
    model_kinds = MODEL_KINDS if args.model_kind == "all" else [args.model_kind]
    ort, onnx_status = load_onnxruntime()

    print("=" * 80)
    print("Jetson Baseline Benchmark".center(80))
    print("=" * 80)
    print(f"Models: {', '.join(model_kinds)}")
    print(f"Feature kind: {args.feature_kind}")
    print(f"Input shape: {input_shape} ({input_shape_source})")
    print(f"Thermometer bins: {args.thermometer_bins}")
    print(f"Device: {device}")
    print(f"Threads: {args.num_threads}")
    print(f"ONNX export/runtime: {onnx_status}")

    results = []
    for model_kind in model_kinds:
        print(f"\nBenchmarking {model_kind}...")
        model_results = benchmark_model_kind(args, model_kind, input_shape, device, ort=ort)
        results.extend(model_results)

    print_results(results, args.batch_size, input_shape)
    csv_path = save_results(results, args, input_shape)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
