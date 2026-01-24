"""
Fused 모델로 detections_ref.bin 생성 (로컬 yolov5 레포 사용)

C 코드와 동일한 HW 출력 포맷 (바이너리)
"""

import sys
import struct
from pathlib import Path

# 로컬 yolov5 레포 사용
YOLOV5_REPO = Path("/Users/kinghong/Desktop/yolov5")
sys.path.insert(0, str(YOLOV5_REPO))

import torch
import numpy as np
from PIL import Image

YOLOV5N_ROOT = Path(__file__).resolve().parents[1]
IMG_PATH = YOLOV5N_ROOT / "data" / "image" / "zidane.jpg"
WEIGHTS_PATH = YOLOV5N_ROOT / "assets" / "yolov5n.pt"
OUT_BIN_PATH = YOLOV5N_ROOT / "data" / "output" / "ref" / "detections.bin"

CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 640


def preprocess_image(img_path, size=640):
    """C 코드와 동일한 전처리 - PIL Image 반환"""
    img = Image.open(img_path).convert('RGB')
    original_w, original_h = img.size
    
    scale = min(size / original_w, size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    img_padded = Image.new('RGB', (size, size), (114, 114, 114))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    return img_padded  # PIL Image 반환


def main():
    print(f"Loading model from local yolov5 repo...")
    
    # 로컬 yolov5의 hubconf.py 사용
    model = torch.hub.load(str(YOLOV5_REPO), 'custom', path=str(WEIGHTS_PATH), 
                           source='local', trust_repo=True)
    
    # torch.hub.load는 자동으로 fuse()를 호출함
    print("Model loaded (auto-fused)")
    
    model.conf = CONF_THRES
    model.iou = IOU_THRES
    model.eval()
    
    # 이미지 전처리 (PIL Image)
    print(f"Preprocessing: {IMG_PATH.name}")
    img = preprocess_image(IMG_PATH, IMG_SIZE)
    print(f"Input size: {img.size}")
    
    # 추론 (AutoShape이 알아서 처리)
    print("Running inference...")
    results = model(img, size=IMG_SIZE)
    
    # 결과 파싱
    preds = results.xyxy[0].cpu().numpy()
    
    rows = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls_id = pred
        # xyxy → xywh (픽셀 좌표)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1)
        h = (y2 - y1)
        rows.append((int(cls_id), conf, cx, cy, w, h))
    
    rows.sort(key=lambda r: r[1], reverse=True)
    
    # HW 출력용 바이너리 파일로 저장
    OUT_BIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_BIN_PATH.open("wb") as f:
        # 1. detection 개수 (1 byte)
        count = min(len(rows), 255)
        f.write(struct.pack('B', count))
        
        # 2. 각 detection을 hw_detection_t 형식으로 저장
        # struct: uint16 x, y, w, h (8 bytes) + uint8 class_id, confidence, reserved[2] (4 bytes) = 12 bytes
        for i in range(count):
            cls_id, conf, cx, cy, w, h = rows[i]
            hw_x = int(cx)
            hw_y = int(cy)
            hw_w = int(w)
            hw_h = int(h)
            hw_conf = int(conf * 255)
            # pack: H=uint16, B=uint8, little-endian, 12 bytes total
            f.write(struct.pack('<HHHHBBBB', 
                                hw_x, hw_y, hw_w, hw_h, 
                                cls_id, hw_conf, 0, 0))
    
    print(f"\nTotal detections: {len(rows)}")
    print(f"Saved to: {OUT_BIN_PATH} ({1 + count * 12} bytes)")
    
    if rows:
        print("\nTop 5 detections:")
        for i, (cls_id, conf, cx, cy, w, h) in enumerate(rows[:5]):
            print(f"  {i+1}. class={cls_id}, conf={conf:.4f}, pos=({cx:.0f},{cy:.0f})")


if __name__ == "__main__":
    main()
