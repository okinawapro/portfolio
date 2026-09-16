"""
generate_zimage.py
==================
Proof-of-concept local inference script for Z-Image-Turbo.

Hardware target : 64 GB RAM / 12 GB VRAM / RTX 3500
Strategy        : enable_model_cpu_offload() -- weights live in RAM and are
                  moved to GPU only for each sub-module's forward pass, so the
                  full 6B-parameter model never has to fit in VRAM at once.
                  Do NOT call pipeline.to("cuda"); cpu_offload manages device
                  placement automatically via accelerate hooks.

No GUI, no web server, no cloud inference, no telemetry, no runtime downloads.

Usage examples
--------------
# Use built-in demo prompt, defaults (1024x1024, 9 steps, random seed):
    python generate_zimage.py

# Custom prompt:
    python generate_zimage.py "a red fox in a snowy forest, oil painting"

# Custom resolution and fixed seed:
    python generate_zimage.py "cyberpunk city at night" --width 768 --height 1344 --seed 42

# All options:
    python generate_zimage.py --help
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency guard -- fail early with an actionable message
# ---------------------------------------------------------------------------

def _require(module_name, install_hint):
    """Import a module by name and exit with a clear message if missing."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(
            f"\n[ERROR] Required package {module_name!r} is not installed.\n"
            f"        Fix: {install_hint}\n",
            file=sys.stderr,
        )
        sys.exit(1)


_require("torch",     "pip install torch --index-url https://download.pytorch.org/whl/cu121")
_require("diffusers", "pip install git+https://github.com/huggingface/diffusers")
_require("PIL",       "pip install pillow")
_require("psutil",    "pip install psutil")

import torch
import psutil
from PIL import Image  # noqa: F401 (imported to confirm PIL is healthy)

try:
    from diffusers import ZImagePipeline
except ImportError as exc:
    print(
        "\n[ERROR] 'ZImagePipeline' was not found in the installed diffusers version.\n"
        "        Z-Image-Turbo requires diffusers >= 0.36.0.dev0 (installed from source).\n"
        "        Fix: pip install git+https://github.com/huggingface/diffusers\n"
        f"        Original error: {exc}\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Optional: pynvml for live VRAM stats (nvidia-ml-py package).
# Falls back to torch.cuda.memory_allocated if unavailable.
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR  = SCRIPT_DIR / "model" / "Z-Image-Turbo"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ---------------------------------------------------------------------------
# Memory / hardware utilities
# ---------------------------------------------------------------------------

def _ram_gb() -> float:
    """Return currently used system RAM in GB."""
    return psutil.virtual_memory().used / 1024 ** 3


def _vram_mb(device_index: int = 0):
    """Return currently used VRAM in MB, or None when unavailable."""
    if _NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 ** 2
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_index) / 1024 ** 2
    return None


def _print_stats(label: str) -> dict:
    """Print a one-line memory snapshot and return it as a dict."""
    ram  = _ram_gb()
    vram = _vram_mb()
    vram_str = f"{vram:.0f} MB" if vram is not None else "N/A"
    print(f"[stats] {label:<30}  RAM={ram:.1f} GB   VRAM={vram_str}")
    return {"ram_gb": round(ram, 2), "vram_mb": round(vram, 1) if vram else None}


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def _check_cuda() -> None:
    """Exit with a clear message if CUDA is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n[ERROR] CUDA is not available.\n"
            "        Ensure:\n"
            "          1. An NVIDIA GPU is present (RTX 3500 expected).\n"
            "          2. A CUDA-enabled PyTorch build is installed.\n"
            "          3. The NVIDIA driver is up to date.\n"
            "        Quick check: python -c \"import torch; print(torch.cuda.is_available())\"\n"
        )
    name  = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[info ] CUDA device : {name}  ({total:.1f} GB total VRAM)")


def _validate_model_dir(path: Path) -> None:
    """Raise a descriptive FileNotFoundError if the model directory is missing or incomplete."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Model directory not found: {path}\n"
            "        Expected layout:\n"
            "          ./model/Z-Image-Turbo/model_index.json\n"
            "          ./model/Z-Image-Turbo/transformer/\n"
            "          ./model/Z-Image-Turbo/vae/  ... etc.\n"
            "        Make sure the model weights are present before running.\n"
        )
    required = [
        "model_index.json", "scheduler", "text_encoder",
        "tokenizer", "transformer", "vae",
    ]
    missing = [e for e in required if not (path / e).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n[ERROR] Model directory is incomplete: {path}\n"
            f"        Missing entries: {missing}\n"
            "        Re-download or verify the model files.\n"
        )


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(model_dir: Path) -> "ZImagePipeline":
    """
    Load ZImagePipeline from a local directory with CPU offloading enabled.

    Design decisions
    ----------------
    local_files_only=True
        Prevents any accidental HuggingFace Hub network call (metadata
        refresh, config download, etc.).  Raises OSError if files are absent.

    low_cpu_mem_usage=True
        Allocates tensors directly in the target dtype, avoiding a transient
        double-RAM peak during model construction (~12 GB saved vs. default).

    torch_dtype=bfloat16
        Halves VRAM relative to float32.  bfloat16 is numerically stable on
        Ampere (RTX 3500) and matches the dtype used during Turbo training.

    enable_model_cpu_offload()
        accelerate registers forward hooks: each sub-module (text_encoder,
        transformer, vae) is moved to GPU just before its forward pass, then
        returned to CPU immediately after.  Peak VRAM stays within a single
        module's footprint rather than the entire 6B model simultaneously.

    IMPORTANT: Do NOT call pipe.to("cuda") after enable_model_cpu_offload().
        That call would override the hooks and move ALL weights to GPU at once,
        risking OOM on a 12 GB card with a 6B-parameter bfloat16 model.
    """
    print(f"\n[info ] Loading pipeline from: {model_dir}")
    t0 = time.perf_counter()

    try:
        pipe = ZImagePipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,   # <-- no network calls ever
        )
    except OSError as exc:
        raise OSError(
            f"\n[ERROR] Failed to load pipeline from: {model_dir}\n"
            "        Possible causes:\n"
            "          - Weight files missing or corrupted.\n"
            "          - model_index.json references a class not present in\n"
            "            the installed diffusers version.\n"
            f"        Original error: {exc}\n"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "\n[ERROR] Unexpected error while loading the pipeline.\n"
            f"        Original error: {exc}\n"
            f"        Traceback:\n{traceback.format_exc()}"
        ) from exc

    # Must be called INSTEAD of pipe.to("cuda").
    pipe.enable_model_cpu_offload()

    elapsed = time.perf_counter() - t0
    print(f"[info ] Pipeline loaded in {elapsed:.1f}s  (cpu_offload active)")
    return pipe


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_image(
    pipe: "ZImagePipeline",
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,
    seed=None,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Run one inference pass and write a PNG + JSON sidecar to output_dir.

    Parameters
    ----------
    pipe                 : Loaded ZImagePipeline with cpu_offload active.
    prompt               : Text prompt string.
    width / height       : Output image dimensions (default 1024x1024).
    num_inference_steps  : Scheduler steps; 9 -> 8 actual DiT forwards.
    guidance_scale       : CFG scale; 0.0 disables guidance (correct for Turbo).
    seed                 : Integer seed for reproducibility; None = random.
    output_dir           : Directory to write PNG and JSON files.

    Returns
    -------
    (png_path, metadata_dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose a random seed when none is specified so every run is unique.
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31, (1,)).item())

    # Generator must target "cuda" to correctly seed the GPU RNG.
    generator = torch.Generator(device="cuda").manual_seed(seed)

    prompt_preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
    print(f"\n[info ] Generating image...")
    print(f"         prompt          : {prompt_preview}")
    print(f"         resolution      : {width}x{height}")
    print(f"         steps           : {num_inference_steps}  -> {num_inference_steps - 1} DiT forwards")
    print(f"         guidance_scale  : {guidance_scale}")
    print(f"         seed            : {seed}")

    stats_before = _print_stats("before generation")
    t0 = time.perf_counter()

    try:
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise MemoryError(
            "\n[ERROR] CUDA out-of-memory during inference.\n"
            "        Suggestions:\n"
            "          - Lower --width / --height (try 768x768 or 512x512).\n"
            "          - Close other GPU-intensive applications.\n"
            "          - Verify enable_model_cpu_offload() is active and that\n"
            "            pipe.to('cuda') was NOT called anywhere.\n"
            f"        Original error: {exc}\n"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "\n[ERROR] Inference failed unexpectedly.\n"
            f"        Original error: {exc}\n"
            f"        Traceback:\n{traceback.format_exc()}"
        ) from exc

    elapsed = time.perf_counter() - t0
    stats_after = _print_stats("after generation")
    print(f"[info ] Done in {elapsed:.2f}s")

    image = result.images[0]

    # Filename: UTC timestamp + seed for easy sorting and reproducibility.
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"zimage_{ts}_seed{seed}"
    png_path  = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    image.save(png_path, format="PNG", optimize=False)

    metadata = {
        "model"               : "Z-Image-Turbo",
        "model_path"          : str(MODEL_DIR),
        "timestamp_utc"       : ts,
        "seed"                : seed,
        "prompt"              : prompt,
        "width"               : width,
        "height"              : height,
        "num_inference_steps" : num_inference_steps,
        "dit_forward_passes"  : num_inference_steps - 1,
        "guidance_scale"      : guidance_scale,
        "torch_dtype"         : "bfloat16",
        "device_strategy"     : "enable_model_cpu_offload",
        "generation_time_s"   : round(elapsed, 3),
        "output_png"          : str(png_path),
        "memory": {
            "before": stats_before,
            "after" : stats_after,
        },
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[done ] Image saved : {png_path}")
    print(f"[done ] Metadata    : {json_path}")
    return png_path, metadata


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_zimage",
        description=(
            "Local Z-Image-Turbo inference -- no network, no GUI, no web server.\n"
            "Loads the model from ./model/Z-Image-Turbo and writes PNG + JSON to ./output/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=(
            "A serene Japanese garden at golden hour, koi pond with lily pads, "
            "stone lanterns, cherry blossom petals drifting on water, "
            "cinematic depth of field, hyperrealistic photography, 8K."
        ),
        help="Text prompt for image generation (default: built-in demo prompt).",
    )
    p.add_argument(
        "--width", type=int, default=1024,
        help="Output image width in pixels (default: 1024).",
    )
    p.add_argument(
        "--height", type=int, default=1024,
        help="Output image height in pixels (default: 1024).",
    )
    p.add_argument(
        "--steps", dest="num_inference_steps", type=int, default=9,
        help="Scheduler steps (default: 9, which equals 8 DiT forwards).",
    )
    p.add_argument(
        "--guidance-scale", type=float, default=0.0,
        help="CFG guidance scale (default: 0.0 -- disabled, recommended for Turbo).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Integer seed for reproducibility (default: random).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Directory to write PNG + JSON output (default: {OUTPUT_DIR}).",
    )
    p.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR,
        help=f"Local Z-Image-Turbo model directory (default: {MODEL_DIR}).",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    print("=" * 60)
    print("  Z-Image-Turbo  --  local inference PoC")
    print("=" * 60)

    # Pre-flight checks
    try:
        _check_cuda()
        _validate_model_dir(args.model_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    _print_stats("startup")

    # Load pipeline (cpu_offload, no .to("cuda"))
    try:
        pipe = load_pipeline(args.model_dir)
    except (OSError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    _print_stats("after model load")

    # Generate
    try:
        generate_image(
            pipe,
            args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    except (MemoryError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[info ] Interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

