"""
Layer 0~9 전체 검증용 테스트 벡터 생성
이미지 입력 → Layer 0~9까지 각 레이어 출력 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--img", help="입력 이미지 경로 (없으면 랜덤 생성)")
    ap.add_argument("--size", type=int, default=64, help="이미지 리사이즈 크기 (정사각형)")
    ap.add_argument("--out", default="tests/test_vectors_layer0_9.h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.pt).model

    # 입력 이미지 준비
    if args.img:
        img = Image.open(args.img).convert('RGB')
        # 너무 크면 C 레퍼런스가 오래 걸려서 작게 줄여서 테스트
        img = img.resize((args.size, args.size))
        img_np = np.array(img, dtype=np.float32) / 255.0  # 0~1 정규화
        # (H, W, C) -> (1, C, H, W)
        x = torch.from_numpy(img_np.transpose(2, 0, 1)[None, :, :, :])
    else:
        rng = np.random.default_rng(args.seed)
        # 랜덤 이미지: (1, 3, size, size)
        x = torch.from_numpy(rng.random((1, 3, args.size, args.size), dtype=np.float32))

    # Layer 0~9까지 forward하면서 각 레이어 출력 저장
    outputs = []
    with torch.no_grad():
        y = x
        for i in range(10):
            y = model.model[i](y)
            outputs.append(y.cpu().numpy().astype(np.float32))
            print(f"Layer {i} 출력 shape: {y.shape}")

    # C 헤더로 저장
    out_path = Path(args.out).expanduser().resolve()

    def dump_array(f, name: str, arr: np.ndarray) -> None:
        flat = arr.reshape(-1)
        f.write(f"static const float {name}[{flat.size}] = {{\n")
        for i, v in enumerate(flat.tolist()):
            f.write(f"{v:.8e}f,")
            if (i + 1) % 8 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n};\n\n")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("#ifndef TEST_VECTORS_LAYER0_9_H\n")
        f.write("#define TEST_VECTORS_LAYER0_9_H\n\n")
        f.write("// 자동 생성됨 (Layer 0~9 전체 검증)\n\n")

        # 입력
        f.write(f"#define TV_L0_9_X_N {x.shape[0]}\n")
        f.write(f"#define TV_L0_9_X_C {x.shape[1]}\n")
        f.write(f"#define TV_L0_9_X_H {x.shape[2]}\n")
        f.write(f"#define TV_L0_9_X_W {x.shape[3]}\n\n")

        dump_array(f, "tv_l0_9_x", x.numpy().astype(np.float32))

        # 각 레이어 출력
        for i, out in enumerate(outputs):
            f.write(f"// Layer {i} 출력\n")
            f.write(f"#define TV_L{i}_N {out.shape[0]}\n")
            f.write(f"#define TV_L{i}_C {out.shape[1]}\n")
            f.write(f"#define TV_L{i}_H {out.shape[2]}\n")
            f.write(f"#define TV_L{i}_W {out.shape[3]}\n\n")
            dump_array(f, f"tv_l{i}_out", out)

        f.write("#endif // TEST_VECTORS_LAYER0_9_H\n")

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
