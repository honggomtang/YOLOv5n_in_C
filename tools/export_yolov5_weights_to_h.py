"""
export_yolov5_weights_to_h.py

Extract weights/biases from a PyTorch .pt file (e.g., yolov5n.pt) and emit a C header (weights.h)
containing float arrays that can be used in embedded C/C++ projects (e.g., Vitis / MicroBlaze).

This script focuses on exporting the model parameters (state_dict).
It exports ALL tensors found in the state_dict. In YOLOv5, most layers have:
  - *.weight (conv weights)
  - *.bias (bias)
  - *.running_mean / *.running_var (BatchNorm stats)
  - *.num_batches_tracked (integer; will be exported as uint64_t)

Usage example:
  python3 export_yolov5_weights_to_h.py --pt yolov5n.pt --out weights.h
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple


def _sanitize_c_identifier(name: str) -> str:
    """
    Turn a PyTorch parameter key into a valid-ish C identifier.
    Example: "model.0.conv.weight" -> "model_0_conv_weight"
    """
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "param"
    if s[0].isdigit():
        s = "p_" + s
    return s


def _load_state_dict(obj: Any) -> Dict[str, Any]:
    """
    YOLOv5 .pt can store:
      - a full model object
      - a dict with keys like 'model', 'ema', etc.
      - a plain state_dict
    We try to find a state_dict robustly.
    """
    # Plain state_dict case
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # If values look like tensors, this might already be a state_dict.
        # But YOLOv5 checkpoints often have 'model'/'ema' entries.
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj:
            m = obj["model"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
            if isinstance(m, dict):
                # Sometimes nested state dict
                return m
        if "ema" in obj:
            m = obj["ema"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
            if isinstance(m, dict):
                return m

        # Heuristic: treat as state_dict if many keys end with weight/bias/running_*
        suffix_hits = 0
        for k in obj.keys():
            if any(k.endswith(s) for s in (".weight", ".bias", ".running_mean", ".running_var")):
                suffix_hits += 1
        if suffix_hits >= 3:
            return obj

    # Full model object case
    if hasattr(obj, "state_dict"):
        return obj.state_dict()

    raise ValueError(
        "Could not locate a state_dict inside the .pt file. "
        "Try exporting a checkpoint that contains model parameters."
    )


def _tensor_to_c_array(t, dtype: str) -> Tuple[str, str, int]:
    """
    Convert a torch tensor to:
      - c_type: "float" or "uint64_t"
      - initializer string (comma-separated)
      - element count
    """
    import numpy as np

    # Move to CPU, detach, convert to numpy
    tn = t.detach().cpu().numpy()

    if dtype == "float":
        tn = tn.astype(np.float32, copy=False).reshape(-1)
        elems = tn.size
        # Use scientific notation; ensure "f" suffix for float literals
        init = ", ".join(f"{x:.8e}f" for x in tn.tolist())
        return "float", init, int(elems)

    if dtype == "u64":
        tn = tn.astype(np.uint64, copy=False).reshape(-1)
        elems = tn.size
        init = ", ".join(str(int(x)) + "ULL" for x in tn.tolist())
        return "uint64_t", init, int(elems)

    raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to yolov5n.pt (or any PyTorch .pt checkpoint)")
    ap.add_argument("--out", default="assets/weights.h", help="Output header path (default: assets/weights.h)")
    ap.add_argument(
        "--only",
        default="",
        help="Optional substring filter: export only keys containing this string (e.g. '.weight' or '.bias')",
    )
    ap.add_argument(
        "--trust-pickle",
        action="store_true",
        help=(
            "If set, load the checkpoint with torch.load(weights_only=False). "
            "ONLY use this if you trust the source of the .pt file (pickle can execute code)."
        ),
    )
    ap.add_argument(
        "--max",
        type=int,
        default=0,
        help="Optional safety limit: maximum number of parameters to export (0 = no limit)",
    )
    args = ap.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not pt_path.exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    # Lazy import torch so the script can show a clean error if torch isn't installed.
    import torch

    # PyTorch 2.6+ defaults weights_only=True, which often fails for YOLOv5 checkpoints
    # that include a pickled Model object (models.yolo.Model). We first try weights_only=True
    # (safer), and if the user explicitly trusts the checkpoint, we retry with weights_only=False.
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if args.trust_pickle:
            try:
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            except ModuleNotFoundError as mnfe:
                raise ModuleNotFoundError(
                    "This checkpoint requires YOLOv5 code (e.g., 'models.yolo.Model') to be importable.\n"
                    "Fix options:\n"
                    "  1) Run this script inside a cloned ultralytics/yolov5 repo, OR\n"
                    "  2) Install a package that provides YOLOv5 modules, then retry.\n"
                    f"Original error: {mnfe}"
                ) from mnfe
        else:
            raise RuntimeError(
                "Failed to load .pt with torch.load(..., weights_only=True).\n"
                "This is common for YOLOv5 checkpoints on PyTorch 2.6+.\n\n"
                "If (and ONLY if) you trust the source of the .pt file, retry with:\n"
                "  python export_yolov5_weights_to_h.py --pt yolov5n.pt --out weights.h --trust-pickle\n\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    state = _load_state_dict(ckpt)

    # Sort keys for stable output
    keys = sorted(state.keys())
    if args.only:
        keys = [k for k in keys if args.only in k]

    if args.max and len(keys) > args.max:
        keys = keys[: args.max]

    # Build header content
    lines = []
    guard = "WEIGHTS_H_"
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("// Auto-generated by export_yolov5_weights_to_h.py")
    lines.append(f"// Source: {pt_path.name}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    exported = 0
    for k in keys:
        v = state[k]
        # Skip non-tensors
        if not hasattr(v, "detach"):
            continue

        # Decide dtype export
        # Most weights/biases are float32/float16. Batch counters are integer.
        is_int = str(v.dtype).startswith("torch.int") or str(v.dtype).startswith("torch.uint") or str(v.dtype) in (
            "torch.long",
            "torch.int64",
        )

        c_name = _sanitize_c_identifier(k)
        shape = tuple(int(x) for x in v.shape)

        if is_int:
            c_type, init, n = _tensor_to_c_array(v, "u64")
        else:
            # Convert any float type to float32 for embedded simplicity
            v = v.float()
            c_type, init, n = _tensor_to_c_array(v, "float")

        lines.append(f"// {k}  shape={shape}  elems={n}")
        lines.append(f"static const {c_type} {c_name}[{n}] = {{ {init} }};")
        lines.append(f"#define {c_name.upper()}_ELEMS ({n})")
        if len(shape) > 0:
            # Export shape macros too (helps reconstruct Conv weights etc.)
            for i, dim in enumerate(shape):
                lines.append(f"#define {c_name.upper()}_DIM{i} ({dim})")
        lines.append("")
        exported += 1

    lines.append(f"// Exported tensors: {exported}")
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path} (tensors={exported})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

