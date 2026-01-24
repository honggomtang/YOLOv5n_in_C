# YOLOv5n Pure C Implementation

외부 라이브러리 없이 순수 C로 구현한 YOLOv5n (nano) 객체 탐지 추론 엔진.

## 프로젝트 목표

**최종 목표**: MicroBlaze V (RISC-V) 프로세서가 구현된 FPGA에서 실행 가능한 YOLOv5n 추론 엔진

**제약 조건**:
- 외부 라이브러리 사용 불가 (OpenCV, OpenBLAS 등)
- Bare-metal / 임베디드 환경 타겟
- 순수 C 언어만 사용

**현재 상태**: Python YOLOv5n과 100% 동일한 추론 결과 출력 완료

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
│   │   ├── decode.c/h        # Anchor-based Decode + hw_detection_t 정의
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
│       ├── detections.bin     # C 결과 (HW 바이너리 포맷)
│       ├── detections.txt     # C 결과 (텍스트)
│       ├── detections.jpg     # C 결과 시각화
│       └── ref/               # Python 참조 결과
│           ├── detections.bin
│           ├── detections.txt
│           └── detections.jpg
│
├── tools/                     # Python 도구
│   ├── export_weights_to_bin.py    # PyTorch → weights.bin 변환
│   ├── preprocess_image_to_bin.py  # 이미지 전처리
│   ├── run_python_yolov5n_fused.py # Python 참조 출력 생성
│   ├── decode_detections.py        # bin → txt 변환 + 시각화
│   └── gen_test_vectors.py         # 테스트 벡터 생성
│
└── tests/                     # 단위 테스트
    ├── test_*.c               # 각 블록별 테스트
    └── test_vectors_*.h       # 테스트 벡터
```

---

## HW 출력 포맷

FPGA에서 호스트로 전송하는 **바이너리 포맷** (detections.bin):

```c
// 12 bytes per detection
typedef struct __attribute__((packed)) {
    uint16_t x, y, w, h;   // 픽셀 좌표 (중심점, 크기)
    uint8_t  class_id;     // 클래스 ID (0~79)
    uint8_t  confidence;   // 신뢰도 (0~255, conf*255)
    uint8_t  reserved[2];  // 정렬용
} hw_detection_t;

// 파일 구조:
// [1 byte]   num_detections (최대 255)
// [12 bytes] detection[0]
// [12 bytes] detection[1]
// ...
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
Decoded: 19 detections
After NMS: 3 detections
Saved to data/output/detections.bin (37 bytes)
```

---

## 전체 워크플로우

### 1. 가중치 준비

```bash
cd /path/to/yolov5
source .venv/bin/activate
python /path/to/yolov5n/tools/export_weights_to_bin.py --classic
```

### 2. 이미지 전처리

```bash
python tools/preprocess_image_to_bin.py \
    --img data/image/zidane.jpg \
    --out data/input/preprocessed_image.bin
```

입력 이미지 처리:
1. RGB 변환
2. 종횡비 유지 리사이즈 (letterbox)
3. 114,114,114 패딩
4. 0-1 정규화
5. NCHW 형식으로 저장

### 3. C 추론 실행

```bash
./main
# 출력: data/output/detections.bin
```

### 4. Python 참조 생성 (선택)

```bash
cd /path/to/yolov5
source .venv/bin/activate
python /path/to/yolov5n/tools/run_python_yolov5n_fused.py
# 출력: data/output/ref/detections.bin
```

### 5. 결과 디코딩 및 시각화

```bash
# C 결과만
python tools/decode_detections.py

# Python 참조만
python tools/decode_detections.py --ref

# 둘 다 비교
python tools/decode_detections.py --compare
```

출력 예시:
```
=== Comparison ===
C detections:   3
Ref detections: 3

Top detections comparison:
C Result                                 | Python Reference                        
-------------------------------------------------------------------------------------
person 0.800 (474,328)                   | person 0.800 (474,328)                  
person 0.388 (218,365)                   | person 0.388 (218,365)                  
tie 0.267 (235,426)                      | tie 0.267 (235,426)                     
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

## 테스트

### 블록별 단위 테스트

```bash
# 테스트 벡터 생성 (yolov5 venv 환경에서)
cd /path/to/yolov5
source .venv/bin/activate
python /path/to/yolov5n/tools/gen_test_vectors.py --all

# 테스트 빌드 및 실행
cd /path/to/yolov5n
gcc -o tests/test_conv tests/test_conv.c csrc/blocks/conv.c csrc/operations/conv2d.c csrc/operations/silu.c csrc/utils/weights_loader.c -I. -Icsrc -lm -std=c99 -O2
./tests/test_conv
```

| 테스트 | 블록 | 검증 내용 |
|--------|------|----------|
| `test_conv` | Conv | Layer 0 (Conv 6x6 s2 + SiLU) |
| `test_c3` | C3 | Layer 2 (C3, n=1, shortcut=True) |
| `test_sppf` | SPPF | Layer 9 (MaxPool k=5 x3) |
| `test_detect` | Detect | Layer 24 (1x1 Conv x3 스케일) |
| `test_decode` | Decode | Anchor-based bbox 디코딩 |
| `test_nms` | NMS | Non-Maximum Suppression |
| `test_upsample` | Upsample | Nearest Neighbor 2x |

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

- **정밀도**: Python YOLOv5n과 100% 동일한 결과
- **출력 포맷**: HW 친화적인 바이너리 (12 bytes/detection)
- **메모리**: 동적 할당 사용 (malloc/free)
- **플랫폼**: macOS/Linux에서 테스트됨, FPGA 포팅 예정
