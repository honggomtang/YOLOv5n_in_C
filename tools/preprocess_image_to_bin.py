"""
이미지 전처리 (letterbox + normalize) → C용 .bin 파일 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import struct


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess image for YOLOv5n inference")
    ap.add_argument("--img", required=True, help="입력 이미지 경로")
    ap.add_argument("--out", required=True, help="출력 .bin 파일 경로")
    ap.add_argument("--size", type=int, default=640, help="이미지 리사이즈 크기")
    ap.add_argument("--quiet", action="store_true", help="로그 출력 비활성화")
    args = ap.parse_args()

    # 이미지 로드 및 전처리
    img = Image.open(args.img).convert('RGB')
    original_w, original_h = img.size
    
    # 리사이즈 (비율 유지)
    scale = min(args.size / original_w, args.size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 패딩 추가 (정사각형으로 만들기)
    img_padded = Image.new('RGB', (args.size, args.size), (114, 114, 114))
    paste_x = (args.size - new_w) // 2
    paste_y = (args.size - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    # Numpy 배열로 변환 및 정규화
    img_np = np.array(img_padded, dtype=np.float32) / 255.0
    img_nchw = img_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W) NCHW
    
    if not args.quiet:
        print(f"Original size: {original_w}x{original_h}")
        print(f"Resized size: {new_w}x{new_h}")
        print(f"Padded size: {args.size}x{args.size}")
        print(f"Image array shape: {img_nchw.shape}")

    # .bin 파일로 저장
    out_path = Path(args.out).expanduser().resolve()
    
    with out_path.open("wb") as f:
        # 헤더: 원본 이미지 정보 (24 bytes)
        f.write(struct.pack("I", original_w))
        f.write(struct.pack("I", original_h))
        f.write(struct.pack("f", scale))
        f.write(struct.pack("I", paste_x))
        f.write(struct.pack("I", paste_y))
        f.write(struct.pack("I", args.size))
        
        # 이미지 데이터 (C, H, W) float32
        f.write(img_nchw.astype(np.float32).tobytes())
    
    if not args.quiet:
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nWrote: {out_path}")
        print(f"File size: {file_size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
