"""
conv0 테스트 벡터 만들기
입력 x랑 conv0 출력 y를 C 헤더로 뽑아줌
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--out", default="tests/test_vectors_conv0.h")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from ultralytics import YOLO

    rng = np.random.default_rng(args.seed)
    # 0~1 범위 랜덤 입력
    x = rng.random((1, 3, args.h, args.w), dtype=np.float32)

    model = YOLO(args.pt).model
    layer0 = model.model[0]

    with torch.no_grad():
        xt = torch.from_numpy(x)
        yt = layer0(xt)  # conv+bn+act 까지 포함
        y = yt.cpu().numpy().astype(np.float32)

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
        f.write("#ifndef TEST_VECTORS_CONV0_H\n")
        f.write("#define TEST_VECTORS_CONV0_H\n\n")
        f.write("// 자동 생성됨\n\n")
        f.write(f"#define TV_X_N 1\n#define TV_X_C 3\n#define TV_X_H {args.h}\n#define TV_X_W {args.w}\n")
        f.write("// conv0 출력은 (1,16,Hout,Wout)\n")
        f.write("#define TV_Y_C 16\n")
        f.write(f"#define TV_Y_H {y.shape[2]}\n#define TV_Y_W {y.shape[3]}\n\n")
        dump_array(f, "tv_x", x)
        dump_array(f, "tv_y", y)
        f.write("#endif // TEST_VECTORS_CONV0_H\n")

    print(f"Wrote: {out_path}")
    print("주의: tv_y는 conv+bn+silu까지 포함된 결과야")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

