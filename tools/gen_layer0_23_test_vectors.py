"""
Layer 0~23 전체 검증용 테스트 벡터 생성
이미지 입력 → Layer 0~23까지 각 레이어 출력 저장
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
    ap.add_argument("--out", default="tests/test_vectors_layer0_23.h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.pt).model

    # 입력 이미지 준비
    if args.img:
        img = Image.open(args.img).convert('RGB')
        img = img.resize((args.size, args.size))
        img_np = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(img_np.transpose(2, 0, 1)[None, :, :, :])
    else:
        rng = np.random.default_rng(args.seed)
        x = torch.from_numpy(rng.random((1, 3, args.size, args.size), dtype=np.float32))

    # Layer 0~23까지 forward하면서 각 레이어 출력 저장
    # Hook으로 각 레이어 출력 캡처 (Python이 Concat 참조 자동 처리)
    outputs = []
    saved = {}  # 각 레이어 출력 저장
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                saved[layer_idx] = output.clone()
        return hook
    
    handles = []
    for i in range(24):
        handles.append(model.model[i].register_forward_hook(make_hook(i)))
    
    # 실제 forward 실행 (Python이 Concat 참조 자동 처리)
    # 하지만 각 레이어를 순차 실행해야 하므로, Concat 참조를 수동 처리
    # Concat 참조:
    # - Layer 12: [11, 6]
    # - Layer 16: [15, 4]
    # - Layer 19: [18, 13]
    # - Layer 22: [21, 9]
    
    with torch.no_grad():
        y = x
        for i in range(24):
            layer = model.model[i]
            layer_type = type(layer).__name__
            
            if 'Concat' in layer_type:
                # Concat 참조 (정확한 구조)
                if i == 12:
                    # Layer 12: concat([Layer 11, Layer 6])
                    y = torch.cat([saved[11], saved[6]], dim=1)
                elif i == 16:
                    # Layer 16: concat([Layer 15, Layer 4])
                    y = torch.cat([saved[15], saved[4]], dim=1)
                elif i == 19:
                    # Layer 19: concat([Layer 18, Layer 14])
                    # Layer 18: 64채널, Layer 14: 64채널 -> 128채널
                    y = torch.cat([saved[18], saved[14]], dim=1)
                elif i == 22:
                    # Layer 22: concat([Layer 21, Layer 10])
                    # Layer 21: 128채널, Layer 10: 128채널 -> 256채널
                    y = torch.cat([saved[21], saved[10]], dim=1)
                else:
                    raise RuntimeError(f"Unknown Concat layer: {i}")
            else:
                y = layer(y)
            
            # Hook이 캡처한 출력 사용 (없으면 직접 저장)
            if i in saved:
                y = saved[i]
            else:
                saved[i] = y.clone()
            
            outputs.append(y.cpu().numpy().astype(np.float32))
            print(f"Layer {i:2d} ({layer_type:12s}) 출력 shape: {y.shape}")
    
    # Hook 제거
    for h in handles:
        h.remove()

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
        f.write("#ifndef TEST_VECTORS_LAYER0_23_H\n")
        f.write("#define TEST_VECTORS_LAYER0_23_H\n\n")
        f.write("// 자동 생성됨 (Layer 0~23 전체 검증)\n\n")

        # 입력
        f.write(f"#define TV_L0_23_X_N {x.shape[0]}\n")
        f.write(f"#define TV_L0_23_X_C {x.shape[1]}\n")
        f.write(f"#define TV_L0_23_X_H {x.shape[2]}\n")
        f.write(f"#define TV_L0_23_X_W {x.shape[3]}\n\n")

        dump_array(f, "tv_l0_23_x", x.numpy().astype(np.float32))

        # 각 레이어 출력
        for i, out in enumerate(outputs):
            f.write(f"// Layer {i} 출력\n")
            f.write(f"#define TV_L{i}_N {out.shape[0]}\n")
            f.write(f"#define TV_L{i}_C {out.shape[1]}\n")
            f.write(f"#define TV_L{i}_H {out.shape[2]}\n")
            f.write(f"#define TV_L{i}_W {out.shape[3]}\n\n")
            dump_array(f, f"tv_l{i}_out", out)

        f.write("#endif // TEST_VECTORS_LAYER0_23_H\n")

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
