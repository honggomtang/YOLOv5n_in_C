# YOLOv5n Pure C Implementation

외부 라이브러리 없이 순수 C로 구현한 YOLOv5n (nano) 객체 탐지 추론 엔진.

## 프로젝트 목표

**최종 목표**: MicroBlaze V (RISC-V) 프로세서가 구현된 FPGA에서 실행 가능한 YOLOv5n 추론 엔진

**제약 조건**:
- 외부 라이브러리 사용 불가 (OpenCV, OpenBLAS 등)
- Bare-metal / 임베디드 환경 타겟
- 순수 C 언어만 사용

**현재 상태**: Python YOLOv5n (Fused)과 동일한 추론 결과 출력 완료

---

## 모델 구조 (YOLOv5n)

총 25개 레이어로 구성: Backbone (10) + Neck (14) + Detect Head (1)

### Backbone (Layer 0-9)

특징 추출을 담당하는 부분.

| Layer | 연산 | 입력 크기 | 출력 크기 | 설명 |
|-------|------|----------|----------|------|
| 0 | Conv 6×6 s2 | 3×640×640 | 16×320×320 | 초기 특징 추출, stride=2로 해상도 절반 |
| 1 | Conv 3×3 s2 | 16×320×320 | 32×160×160 | 채널 확장 및 다운샘플링 |
| 2 | C3 (n=1) | 32×160×160 | 32×160×160 | Bottleneck 1회, shortcut=True |
| 3 | Conv 3×3 s2 | 32×160×160 | 64×80×80 | 다운샘플링 → P3 스케일 |
| 4 | C3 (n=2) | 64×80×80 | 64×80×80 | Bottleneck 2회, shortcut=True, **Neck 연결점** |
| 5 | Conv 3×3 s2 | 64×80×80 | 128×40×40 | 다운샘플링 → P4 스케일 |
| 6 | C3 (n=3) | 128×40×40 | 128×40×40 | Bottleneck 3회, shortcut=True, **Neck 연결점** |
| 7 | Conv 3×3 s2 | 128×40×40 | 256×20×20 | 다운샘플링 → P5 스케일 |
| 8 | C3 (n=1) | 256×20×20 | 256×20×20 | Bottleneck 1회, shortcut=True |
| 9 | SPPF | 256×20×20 | 256×20×20 | Spatial Pyramid Pooling Fast |

### Neck (Layer 10-23) - PANet 구조

다중 스케일 특징을 융합하는 Feature Pyramid Network.

| Layer | 연산 | 입력 | 출력 크기 | 설명 |
|-------|------|------|----------|------|
| 10 | Conv 1×1 | l9 | 128×20×20 | 채널 축소 |
| 11 | Upsample 2× | l10 | 128×40×40 | 해상도 2배 확대 |
| 12 | Concat | l11 + l6 | 256×40×40 | Skip connection |
| 13 | C3 (n=1) | l12 | 128×40×40 | shortcut=False |
| 14 | Conv 1×1 | l13 | 64×40×40 | 채널 축소 |
| 15 | Upsample 2× | l14 | 64×80×80 | 해상도 2배 확대 |
| 16 | Concat | l15 + l4 | 128×80×80 | Skip connection |
| 17 | C3 (n=1) | l16 | **64×80×80** | shortcut=False, **→ P3 출력** |
| 18 | Conv 3×3 s2 | l17 | 64×40×40 | 다운샘플링 |
| 19 | Concat | l18 + l14 | 128×40×40 | Skip connection |
| 20 | C3 (n=1) | l19 | **128×40×40** | shortcut=False, **→ P4 출력** |
| 21 | Conv 3×3 s2 | l20 | 128×20×20 | 다운샘플링 |
| 22 | Concat | l21 + l10 | 256×20×20 | Skip connection |
| 23 | C3 (n=1) | l22 | **256×20×20** | shortcut=False, **→ P5 출력** |

### Detect Head (Layer 24)

3개 스케일에서 각각 1×1 Conv로 255채널 출력 생성.

| 스케일 | 입력 | Conv | 출력 | 앵커 |
|--------|------|------|------|------|
| P3 (Small) | 64×80×80 | 1×1, 64→255 | 255×80×80 | [10,13], [16,30], [33,23] |
| P4 (Medium) | 128×40×40 | 1×1, 128→255 | 255×40×40 | [30,61], [62,45], [59,119] |
| P5 (Large) | 256×20×20 | 1×1, 256→255 | 255×20×20 | [116,90], [156,198], [373,326] |

**255 = 3 앵커 × 85 (4 bbox + 1 obj + 80 classes)**

---

## 폴더 구조

```
yolov5n/
├── assets/                     # 모델 파일
│   ├── yolov5n.pt             # PyTorch 원본 모델
│   └── weights.bin            # C용 변환된 가중치 (Fused)
│
├── csrc/                       # C 소스 코드
│   ├── main.c                 # 메인 추론 파이프라인
│   │
│   ├── blocks/                # 고수준 블록
│   │   ├── conv.c/h          # Conv 블록 (Conv2D + Bias + SiLU)
│   │   ├── c3.c/h            # C3 블록 (cv1 + cv2 + Bottleneck + cv3)
│   │   ├── sppf.c/h          # SPPF 블록 (Spatial Pyramid Pooling Fast)
│   │   ├── detect.c/h        # Detect Head (1×1 Conv × 3 스케일)
│   │   ├── decode.c/h        # Anchor-based Decode + detection_t 정의
│   │   └── nms.c/h           # Non-Maximum Suppression
│   │
│   ├── operations/            # 저수준 연산
│   │   ├── conv2d.c/h        # 2D Convolution
│   │   ├── silu.c/h          # SiLU 활성화 함수
│   │   ├── bottleneck.c/h    # Bottleneck 모듈
│   │   ├── concat.c/h        # 채널 방향 Concat
│   │   ├── maxpool2d.c/h     # 2D Max Pooling
│   │   └── upsample.c/h      # Nearest Neighbor 2× Upsampling
│   │
│   └── utils/                 # 유틸리티
│       ├── weights_loader.c/h # weights.bin 로더
│       └── image_loader.c/h   # 전처리된 이미지 로더
│
├── data/
│   ├── image/                 # 입력 이미지
│   │   └── zidane.jpg
│   ├── input/                 # 전처리된 입력
│   │   └── preprocessed_image.bin
│   └── output/                # 추론 결과
│       ├── detections.txt     # C 코드 출력
│       └── detections_ref.txt # Python 참조 출력
│
├── tools/                     # Python 도구
│   ├── export_weights_to_bin.py    # PyTorch → weights.bin 변환
│   ├── preprocess_image_to_bin.py  # 이미지 전처리
│   ├── run_python_yolov5n_fused.py # Python 참조 출력 생성
│   └── gen_*.py                    # 테스트 벡터 생성
│
└── tests/                     # 단위 테스트
    ├── test_*.c               # 각 블록별 테스트
    └── test_vectors_*.h       # 테스트 벡터
```

---

## 핵심 기술 사항

### 1. Fused Model 사용

**Conv + BatchNorm → Conv + Bias** 형태로 통합된 가중치 사용.

```
일반 모델: x → Conv → BatchNorm → SiLU → y
Fused 모델: x → Conv+Bias → SiLU → y  (BN이 Conv에 흡수됨)
```

**장점**: 연산량 감소, 코드 단순화, 메모리 절약

### 2. NCHW 데이터 포맷

모든 텐서는 **N×C×H×W** (Batch × Channel × Height × Width) 형태.

```c
// 인덱싱: tensor[n][c][h][w] = data[n*C*H*W + c*H*W + h*W + w]
float value = data[c * H * W + y * W + x];
```

### 3. Anchor-based Detection

각 그리드 셀에서 3개의 앵커 박스 사용.

```
P3 (80×80): stride=8,  앵커 [10,13], [16,30], [33,23]   - 작은 객체
P4 (40×40): stride=16, 앵커 [30,61], [62,45], [59,119]  - 중간 객체
P5 (20×20): stride=32, 앵커 [116,90], [156,198], [373,326] - 큰 객체
```

**디코딩 공식**:
```
cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
cy = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
w = (sigmoid(tw) * 2)² * anchor_w
h = (sigmoid(th) * 2)² * anchor_h
conf = sigmoid(obj) * sigmoid(max_cls)
```

### 4. C3 블록의 Shortcut 설정

| 위치 | Shortcut | 이유 |
|------|----------|------|
| Backbone (Layer 2,4,6,8) | True | 그래디언트 흐름 개선 |
| Neck (Layer 13,17,20,23) | False | Feature 융합에 집중 |

---

## 빌드 및 실행

### 빌드

```bash
cd yolov5n
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2
```

### 실행

```bash
./main
```

### 출력 예시

```
=== YOLOv5n Inference (Fused) ===

Image: 640x640
Weights: 121 tensors

Running inference...
Decoded: 31 detections
After NMS: 3 detections
Saved to data/output/detections.txt
```

---

## 전체 워크플로우

### 1. 가중치 준비

```bash
# PyTorch 모델 → Fused weights.bin 변환
python tools/export_weights_to_bin.py --classic
```

### 2. 이미지 전처리

```bash
# 이미지 → 640×640 정규화된 바이너리
python tools/preprocess_image_to_bin.py
```

입력 이미지 처리:
1. RGB 변환
2. 종횡비 유지 리사이즈 (letterbox)
3. 114,114,114 패딩
4. 0-1 정규화
5. NCHW 형식으로 저장

### 3. 추론 실행

```bash
./main
```

### 4. 결과 확인

`data/output/detections.txt`:
```
# YOLOv5n Detection Results
# Detections: 3
# Format: class_id confidence x y w h

0 0.904610 0.740620 0.515294 0.308028 0.511584
0 0.659625 0.338334 0.570972 0.464188 0.409941
27 0.599798 0.371467 0.661167 0.053645 0.226830
```

### 5. Python 참조와 비교 (선택)

```bash
cd /path/to/yolov5
source .venv/bin/activate
python /path/to/yolov5n/tools/run_python_yolov5n_fused.py
```

---

## 구현된 연산 목록

### Blocks

| 블록 | 함수 | 설명 |
|------|------|------|
| Conv | `conv_block_nchw_f32` | Conv2D + Bias + SiLU |
| C3 | `c3_nchw_f32` | CSP Bottleneck with 3 convolutions |
| SPPF | `sppf_nchw_f32` | Spatial Pyramid Pooling Fast |
| Detect | `detect_nchw_f32` | 3-scale 1×1 Conv head |
| Decode | `decode_nchw_f32` | Anchor-based bbox 디코딩 |
| NMS | `nms` | Non-Maximum Suppression |

### Operations

| 연산 | 함수 | 설명 |
|------|------|------|
| Conv2D | `conv2d_nchw_f32` | 2D Convolution (stride, padding, dilation 지원) |
| SiLU | `silu_nchw_f32` | x * sigmoid(x) 활성화 |
| Bottleneck | `bottleneck_nchw_f32` | 1×1 Conv + 3×3 Conv + optional shortcut |
| Concat | `concat_nchw_f32` | 채널 방향 연결 |
| MaxPool2D | `maxpool2d_nchw_f32` | 최대값 풀링 |
| Upsample | `upsample_nearest2x_nchw_f32` | 2배 Nearest Neighbor 업샘플링 |

---

## 상수 및 하이퍼파라미터

```c
#define INPUT_SIZE      640      // 입력 이미지 크기
#define NUM_CLASSES     80       // COCO 클래스 수
#define CONF_THRESHOLD  0.25f    // Confidence 임계값
#define IOU_THRESHOLD   0.45f    // NMS IoU 임계값
#define MAX_DETECTIONS  300      // 최대 detection 수
```

---

## 참고 사항

- **정밀도**: Python Fused 모델과 소수점 6자리까지 일치
- **메모리**: 동적 할당 사용 (malloc/free)
- **플랫폼**: macOS/Linux에서 테스트됨, FPGA 포팅 예정
