"""
PyTorch .pt 모델을 C용 weights.bin으로 변환 (Fused 모델 지원)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import struct
import numpy as np
import torch


def _load_state_dict(obj: Any) -> Dict[str, Any]:
    """YOLOv5 .pt에서 state_dict 추출 (여러 형식 지원)"""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--out", default="assets/weights.bin")
    ap.add_argument("--trust-pickle", action="store_true", help="PyTorch 2.6+ weights_only=False")
    ap.add_argument(
        "--classic",
        action="store_true",
        help="Anchor-based (Standard YOLOv5n). torch.hub ultralytics/yolov5 custom. "
        "Needs network. Use for detections_ref match (desktop detect.py).",
    )
    args = ap.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    state_dict = None

    if args.classic:
        # Anchor-based YOLOv5n (model.24.m.0/m.1/m.2, 255ch)
        try:
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(pt_path),
                force_reload=False,
                trust_repo=True,
            )
            state_dict = model.state_dict()
            print("Loaded model using torch.hub ultralytics/yolov5 (classic)")
        except Exception as e:
            raise RuntimeError(
                f"Classic export failed (needs network): {e}\n"
                "Run from env with internet, or use export without --classic."
            ) from e

    if state_dict is None:
        # DFL(Ultralytics) 또는 torch.load 폴백
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            state_dict = model.model.state_dict()
            print("Loaded model using ultralytics (DFL)")
        except Exception:
            try:
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=True)
                state_dict = _load_state_dict(ckpt)
            except Exception as e:
                if args.trust_pickle:
                    try:
                        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                        state_dict = _load_state_dict(ckpt)
                    except ModuleNotFoundError as mnfe:
                        raise ModuleNotFoundError(
                            f"Checkpoint requires YOLOv5 code to be importable.\n"
                            f"Options: 1) pip install ultralytics, 2) Run inside yolov5 repo.\n"
                            f"Error: {mnfe}"
                        ) from mnfe
                else:
                    raise RuntimeError(
                        f"Failed to load .pt with weights_only=True (PyTorch 2.6+).\n"
                        f"If trusted, retry with --trust-pickle.\n"
                        f"Error: {type(e).__name__}: {e}"
                    ) from e

    # 바이너리 파일로 저장
    out_path = Path(args.out).expanduser().resolve()
    
    with out_path.open("wb") as f:
        # 헤더: 텐서 개수 (4 bytes)
        num_tensors = len(state_dict)
        f.write(struct.pack("I", num_tensors))
        
        # 각 텐서 저장
        for key, tensor in state_dict.items():
            # 키 이름 (길이 + 문자열)
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("I", len(key_bytes)))
            f.write(key_bytes)
            
            # 텐서 shape (차원 수 + 각 차원 크기)
            shape = tensor.shape
            f.write(struct.pack("I", len(shape)))
            for dim in shape:
                f.write(struct.pack("I", dim))
            
            # 텐서 데이터 (float32)
            data = tensor.cpu().numpy().astype(np.float32)
            f.write(data.tobytes())
    
    print(f"Wrote {num_tensors} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
