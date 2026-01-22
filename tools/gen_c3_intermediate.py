"""
C3 블록 중간 출력 생성 (디버깅용)
각 단계별 출력을 저장해서 C 구현과 비교
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="yolov5n.pt")
    ap.add_argument("--out", default="test_vectors_c3_intermediate.h")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.random((1, 32, args.h, args.w), dtype=np.float32)

    model = YOLO(args.pt).model
    layer2 = model.model[2]

    with torch.no_grad():
        xt = torch.from_numpy(x)
        
        # 단계별 실행
        cv1_out = layer2.cv1(xt)
        bottleneck_out = layer2.m[0](cv1_out)
        cv2_out = layer2.cv2(xt)
        concat_out = torch.cat([bottleneck_out, cv2_out], 1)
        cv3_out = layer2.cv3(concat_out)
        
        # numpy로 변환
        cv1_np = cv1_out.cpu().numpy().astype(np.float32)
        bottleneck_np = bottleneck_out.cpu().numpy().astype(np.float32)
        cv2_np = cv2_out.cpu().numpy().astype(np.float32)
        concat_np = concat_out.cpu().numpy().astype(np.float32)
        cv3_np = cv3_out.cpu().numpy().astype(np.float32)

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
        f.write("#ifndef TEST_VECTORS_C3_INTERMEDIATE_H\n")
        f.write("#define TEST_VECTORS_C3_INTERMEDIATE_H\n\n")
        f.write("// 자동 생성됨 (C3 블록 중간 출력)\n\n")
        
        f.write(f"#define TV_C3_X_N 1\n#define TV_C3_X_C 32\n#define TV_C3_X_H {args.h}\n#define TV_C3_X_W {args.w}\n\n")
        
        f.write(f"// cv1 출력: {cv1_np.shape}\n")
        f.write(f"#define TV_C3_CV1_C {cv1_np.shape[1]}\n")
        f.write(f"#define TV_C3_CV1_H {cv1_np.shape[2]}\n")
        f.write(f"#define TV_C3_CV1_W {cv1_np.shape[3]}\n\n")
        
        f.write(f"// bottleneck 출력: {bottleneck_np.shape}\n")
        f.write(f"#define TV_C3_BOTTLENECK_C {bottleneck_np.shape[1]}\n\n")
        
        f.write(f"// cv2 출력: {cv2_np.shape}\n")
        f.write(f"#define TV_C3_CV2_C {cv2_np.shape[1]}\n\n")
        
        f.write(f"// concat 출력: {concat_np.shape}\n")
        f.write(f"#define TV_C3_CONCAT_C {concat_np.shape[1]}\n\n")
        
        f.write(f"// cv3 출력: {cv3_np.shape}\n")
        f.write(f"#define TV_C3_CV3_C {cv3_np.shape[1]}\n")
        f.write(f"#define TV_C3_CV3_H {cv3_np.shape[2]}\n")
        f.write(f"#define TV_C3_CV3_W {cv3_np.shape[3]}\n\n")
        
        dump_array(f, "tv_c3_x", x)
        dump_array(f, "tv_c3_cv1_out", cv1_np)
        dump_array(f, "tv_c3_bottleneck_out", bottleneck_np)
        dump_array(f, "tv_c3_cv2_out", cv2_np)
        dump_array(f, "tv_c3_concat_out", concat_np)
        dump_array(f, "tv_c3_cv3_out", cv3_np)
        
        f.write("#endif // TEST_VECTORS_C3_INTERMEDIATE_H\n")

    print(f"Wrote: {out_path}")
    print(f"입력: {x.shape}")
    print(f"cv1: {cv1_np.shape}, mean={cv1_np.mean():.6f}, std={cv1_np.std():.6f}")
    print(f"bottleneck: {bottleneck_np.shape}, mean={bottleneck_np.mean():.6f}, std={bottleneck_np.std():.6f}")
    print(f"cv2: {cv2_np.shape}, mean={cv2_np.mean():.6f}, std={cv2_np.std():.6f}")
    print(f"concat: {concat_np.shape}, mean={concat_np.mean():.6f}, std={concat_np.std():.6f}")
    print(f"cv3: {cv3_np.shape}, mean={cv3_np.mean():.6f}, std={cv3_np.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
