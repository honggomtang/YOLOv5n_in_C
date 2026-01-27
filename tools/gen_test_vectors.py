#!/usr/bin/env python3
"""
테스트 벡터 생성 (Fused 모델용)

모든 블록의 테스트 벡터를 한번에 생성합니다.

사용법:
    cd /path/to/yolov5
    source .venv/bin/activate
    python /path/to/yolov5n/tools/gen_test_vectors.py
"""
import os
import sys
from pathlib import Path
import argparse

# YOLOv5 repo 경로 설정 (기본: ~/Desktop/yolov5, 또는 환경변수 YOLOV5_REPO)
YOLOV5_REPO = Path(os.environ.get("YOLOV5_REPO", str(Path.home() / "Desktop" / "yolov5")))
if YOLOV5_REPO.exists():
    sys.path.insert(0, str(YOLOV5_REPO))

import numpy as np
import torch

# 프로젝트 경로
PROJECT_DIR = Path(__file__).parent.parent
TESTS_DIR = PROJECT_DIR / "tests"
WEIGHTS_PATH = PROJECT_DIR / "assets" / "yolov5n.pt"

def to_c_array(arr, name, dtype="float"):
    """numpy 배열을 C 배열 문자열로 변환"""
    flat = arr.flatten().astype(np.float32)
    lines = []
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        line = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(line)
    return f"static const {dtype} {name}[{len(flat)}] = {{\n" + ",\n".join(lines) + "\n};\n"

def gen_conv_vectors(model):
    """Conv 블록 (Layer 0) 테스트 벡터"""
    print("Generating Conv vectors...")
    
    H, W = 64, 64  # 테스트용 작은 크기
    np.random.seed(0)
    x = np.random.rand(1, 3, H, W).astype(np.float32)
    
    # Layer 0 실행
    layer0 = model.model[0]  # Conv
    with torch.no_grad():
        xt = torch.from_numpy(x)
        yt = layer0(xt)
        y = yt.cpu().numpy()
    
    h_out, w_out = y.shape[2], y.shape[3]
    
    content = f"""#ifndef TEST_VECTORS_CONV_H
#define TEST_VECTORS_CONV_H

// 자동 생성됨 (Conv Block - Fused)

#define TV_X_N 1
#define TV_X_C 3
#define TV_X_H {H}
#define TV_X_W {W}

#define TV_Y_C 16
#define TV_Y_H {h_out}
#define TV_Y_W {w_out}

{to_c_array(x, "tv_x")}

{to_c_array(y, "tv_y")}

#endif // TEST_VECTORS_CONV_H
"""
    path = TESTS_DIR / "test_vectors_conv.h"
    path.write_text(content)
    print(f"  Written: {path}")

def gen_c3_vectors(model):
    """C3 블록 (Layer 2) 테스트 벡터"""
    print("Generating C3 vectors...")
    
    H, W = 32, 32  # 테스트용 작은 크기
    np.random.seed(42)
    
    # Layer 2 입력은 Layer 1 출력과 같은 shape: (1, 32, H, W)
    x = np.random.rand(1, 32, H, W).astype(np.float32)
    
    # Layer 2 (C3) 실행
    layer2 = model.model[2]  # C3
    with torch.no_grad():
        xt = torch.from_numpy(x)
        yt = layer2(xt)
        y = yt.cpu().numpy()
    
    content = f"""#ifndef TEST_VECTORS_C3_H
#define TEST_VECTORS_C3_H

// 자동 생성됨 (C3 Block - Fused)

#define TV_C3_X_N 1
#define TV_C3_X_C 32
#define TV_C3_X_H {H}
#define TV_C3_X_W {W}

#define TV_C3_Y_C 32
#define TV_C3_Y_H {H}
#define TV_C3_Y_W {W}

{to_c_array(x, "tv_c3_x")}

{to_c_array(y, "tv_c3_y")}

#endif // TEST_VECTORS_C3_H
"""
    path = TESTS_DIR / "test_vectors_c3.h"
    path.write_text(content)
    print(f"  Written: {path}")

def gen_sppf_vectors(model):
    """SPPF 블록 (Layer 9) 테스트 벡터"""
    print("Generating SPPF vectors...")
    
    H, W = 8, 8  # 테스트용 작은 크기
    np.random.seed(123)
    
    # Layer 9 입력: (1, 256, H, W)
    x = np.random.rand(1, 256, H, W).astype(np.float32)
    
    # Layer 9 (SPPF) 실행
    layer9 = model.model[9]  # SPPF
    with torch.no_grad():
        xt = torch.from_numpy(x)
        yt = layer9(xt)
        y = yt.cpu().numpy()
    
    content = f"""#ifndef TEST_VECTORS_SPPF_H
#define TEST_VECTORS_SPPF_H

// 자동 생성됨 (SPPF Block - Fused)

#define TV_SPPF_X_N 1
#define TV_SPPF_X_C 256
#define TV_SPPF_X_H {H}
#define TV_SPPF_X_W {W}

#define TV_SPPF_Y_C 256
#define TV_SPPF_Y_H {H}
#define TV_SPPF_Y_W {W}

{to_c_array(x, "tv_sppf_x")}

{to_c_array(y, "tv_sppf_y")}

#endif // TEST_VECTORS_SPPF_H
"""
    path = TESTS_DIR / "test_vectors_sppf.h"
    path.write_text(content)
    print(f"  Written: {path}")

def gen_detect_vectors(model):
    """Detect Head (Layer 24) 테스트 벡터"""
    print("Generating Detect vectors...")
    
    np.random.seed(456)
    
    # P3, P4, P5 입력 (작은 크기)
    p3 = np.random.rand(1, 64, 4, 4).astype(np.float32)
    p4 = np.random.rand(1, 128, 2, 2).astype(np.float32)
    p5 = np.random.rand(1, 256, 1, 1).astype(np.float32)
    
    # Detect head
    detect = model.model[24]  # Detect layer
    m0, m1, m2 = detect.m[0], detect.m[1], detect.m[2]
    
    with torch.no_grad():
        p3_out = m0(torch.from_numpy(p3)).numpy()
        p4_out = m1(torch.from_numpy(p4)).numpy()
        p5_out = m2(torch.from_numpy(p5)).numpy()
    
    content = f"""#ifndef TEST_VECTORS_DETECT_H
#define TEST_VECTORS_DETECT_H

// 자동 생성됨 (Detect Head - Anchor-based Fused)

// P3 입력
#define TV_DETECT_P3_C 64
#define TV_DETECT_P3_H 4
#define TV_DETECT_P3_W 4

// P4 입력
#define TV_DETECT_P4_C 128
#define TV_DETECT_P4_H 2
#define TV_DETECT_P4_W 2

// P5 입력
#define TV_DETECT_P5_C 256
#define TV_DETECT_P5_H 1
#define TV_DETECT_P5_W 1

{to_c_array(p3, "tv_detect_p3")}

{to_c_array(p4, "tv_detect_p4")}

{to_c_array(p5, "tv_detect_p5")}

// P3 출력 (255채널)
{to_c_array(p3_out, "tv_detect_p3_out")}

// P4 출력 (255채널)
{to_c_array(p4_out, "tv_detect_p4_out")}

// P5 출력 (255채널)
{to_c_array(p5_out, "tv_detect_p5_out")}

#endif // TEST_VECTORS_DETECT_H
"""
    path = TESTS_DIR / "test_vectors_detect.h"
    path.write_text(content)
    print(f"  Written: {path}")
    
    return p3_out, p4_out, p5_out

def gen_decode_vectors(p3_out, p4_out, p5_out):
    """Decode 테스트 벡터 (Detect 출력 사용)"""
    print("Generating Decode vectors...")
    
    content = f"""#ifndef TEST_VECTORS_DECODE_H
#define TEST_VECTORS_DECODE_H

// 자동 생성됨 (Decode - Anchor-based)

#include "../csrc/blocks/decode.h"

// Detect 출력 크기
#define TV_DECODE_P3_H 4
#define TV_DECODE_P3_W 4
#define TV_DECODE_P4_H 2
#define TV_DECODE_P4_W 2
#define TV_DECODE_P5_H 1
#define TV_DECODE_P5_W 1

#define TV_DECODE_INPUT_SIZE 32
#define TV_DECODE_NUM_CLASSES 80
#define TV_DECODE_CONF_THRESHOLD 0.001f

// 작은 랜덤 입력에서는 유의미한 detection이 거의 없음
#define TV_DECODE_NUM_DETECTIONS 0

static const detection_t tv_decode_detections[] = {{
    {{0, 0, 0, 0, 0, -1}}
}};

// Detect 출력 (decode 입력으로 사용)
{to_c_array(p3_out, "tv_decode_p3")}

{to_c_array(p4_out, "tv_decode_p4")}

{to_c_array(p5_out, "tv_decode_p5")}

#endif // TEST_VECTORS_DECODE_H
"""
    path = TESTS_DIR / "test_vectors_decode.h"
    path.write_text(content)
    print(f"  Written: {path}")

def gen_upsample_vectors():
    """Upsample 테스트 벡터 (모델 불필요)"""
    print("Generating Upsample vectors...")
    
    H, W = 4, 4
    C = 128
    np.random.seed(789)
    
    x = np.random.rand(1, C, H, W).astype(np.float32)
    
    # Nearest neighbor 2x upsample
    y = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
    
    content = f"""#ifndef TEST_VECTORS_UPSAMPLE_H
#define TEST_VECTORS_UPSAMPLE_H

// 자동 생성됨 (Upsample 2x)

#define TV_UPSAMPLE_X_N 1
#define TV_UPSAMPLE_X_C {C}
#define TV_UPSAMPLE_X_H {H}
#define TV_UPSAMPLE_X_W {W}

#define TV_UPSAMPLE_Y_C {C}
#define TV_UPSAMPLE_Y_H {H * 2}
#define TV_UPSAMPLE_Y_W {W * 2}

{to_c_array(x, "tv_upsample_x")}

{to_c_array(y, "tv_upsample_y")}

#endif // TEST_VECTORS_UPSAMPLE_H
"""
    path = TESTS_DIR / "test_vectors_upsample.h"
    path.write_text(content)
    print(f"  Written: {path}")

def load_model():
    """모델 로드 및 내부 모델 반환"""
    print("Loading model...")
    
    # torch.hub.load는 AutoShape이 적용되므로, 내부 모델을 직접 로드
    from models.yolo import DetectionModel
    from models.common import DetectMultiBackend
    
    # DetectMultiBackend로 로드하면 내부 모델에 접근 가능
    model = DetectMultiBackend(str(WEIGHTS_PATH), device=torch.device('cpu'))
    model.model.fuse() if hasattr(model.model, 'fuse') else None
    
    # 내부 DetectionModel 반환
    inner_model = model.model
    inner_model.eval()
    
    print("Model loaded (Fused)\n")
    return inner_model

def main():
    parser = argparse.ArgumentParser(description="Generate test vectors for Fused model")
    parser.add_argument("--all", action="store_true", help="Generate all test vectors")
    parser.add_argument("--conv", action="store_true", help="Generate Conv vectors")
    parser.add_argument("--c3", action="store_true", help="Generate C3 vectors")
    parser.add_argument("--sppf", action="store_true", help="Generate SPPF vectors")
    parser.add_argument("--detect", action="store_true", help="Generate Detect vectors")
    parser.add_argument("--decode", action="store_true", help="Generate Decode vectors")
    parser.add_argument("--upsample", action="store_true", help="Generate Upsample vectors")
    args = parser.parse_args()
    
    # 아무 옵션도 없으면 --all
    if not any([args.conv, args.c3, args.sppf, args.detect, args.decode, args.upsample]):
        args.all = True
    
    print("=== Test Vector Generator (Fused Model) ===\n")
    print(f"YOLOv5 repo: {YOLOV5_REPO}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Output: {TESTS_DIR}\n")
    
    # 모델 로드 (upsample 제외한 모든 경우에 필요)
    model = None
    if args.all or args.conv or args.c3 or args.sppf or args.detect or args.decode:
        model = load_model()
    
    # 각 블록 테스트 벡터 생성
    if args.all or args.conv:
        gen_conv_vectors(model)
    
    if args.all or args.c3:
        gen_c3_vectors(model)
    
    if args.all or args.sppf:
        gen_sppf_vectors(model)
    
    p3_out, p4_out, p5_out = None, None, None
    if args.all or args.detect:
        p3_out, p4_out, p5_out = gen_detect_vectors(model)
    
    if args.all or args.decode:
        if p3_out is None:
            # Detect 먼저 실행
            p3_out, p4_out, p5_out = gen_detect_vectors(model)
        gen_decode_vectors(p3_out, p4_out, p5_out)
    
    if args.all or args.upsample:
        gen_upsample_vectors()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
