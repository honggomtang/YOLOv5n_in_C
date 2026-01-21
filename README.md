
# YOLOv5n `.pt` → C Header (`weights.h`) Exporter

전기전자공학부/임베디드(특히 MicroBlaze V / Vitis) 환경에서 쓰기 쉽도록,
PyTorch 모델 파일(`yolov5n.pt`)의 **가중치(Weights) / 편향(Bias)** 등을 뽑아서
`weights.h` 형태의 C 헤더로 내보내는 도구입니다.

## 1) 준비물

- `yolov5n.pt` 파일을 이 폴더에 넣기
  - 예: `/Users/kinghong/Desktop/yolov5n/yolov5n.pt`
- Python 3 설치 (macOS 기본 `python3` 권장)

## 2) 파이썬 환경 설치(처음 1회)

가장 무난한 방법은 가상환경(venv)입니다.

```bash
cd /Users/kinghong/Desktop/yolov5n
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy
```

> 참고: `torch` 설치가 오래 걸릴 수 있어요. (CPU 버전)

## 3) `weights.h` 생성

### A. 전체 텐서(state_dict 전부) 내보내기

```bash
cd /Users/kinghong/Desktop/yolov5n
source .venv/bin/activate
python export_yolov5_weights_to_h.py --pt yolov5n.pt --out weights.h
```

### B. weight만(또는 bias만) 뽑고 싶을 때 (추천)

`weights.h`가 너무 커질 수 있어서, 처음에는 필요한 것만 뽑는 걸 추천합니다.

- weight만:

```bash
python export_yolov5_weights_to_h.py --pt yolov5n.pt --out weights_weight_only.h --only ".weight"
```

- bias만:

```bash
python export_yolov5_weights_to_h.py --pt yolov5n.pt --out weights_bias_only.h --only ".bias"
```

## 3.5) Conv2D부터 검증(추천 루트)

YOLOv5n 첫 블록은 `Conv2d + BN + SiLU`야.

- conv0: `Conv2d(3 -> 16, k=6, s=2, p=2, bias=False)`
- bn0: `BatchNorm2d(16, eps=1e-3)`
- act0: `SiLU`

### A. 파이썬에서 테스트 벡터 뽑기

```bash
cd /Users/kinghong/Desktop/yolov5n
source .venv/bin/activate
python gen_conv0_test_vectors.py --pt yolov5n.pt --out test_vectors_conv0.h --h 64 --w 64
```

### B. C 레퍼런스 구현으로 비교하기(PC에서 먼저)

`test_conv0.c`는 `weights.h`(모델 파라미터) + `test_vectors_conv0.h`(입력/정답)로
conv0를 돌리고 `max_abs_diff`를 출력해.

예시(맥/리눅스):

```bash
cc -O2 -std=c11 test_conv0.c conv2d_ref.c bn_silu_ref.c -lm -o test_conv0
./test_conv0
```

여기서 `OK` 뜨면, 그 다음부터 MicroBlaze로 포팅하면 돼.

## 4) 출력 형식(헤더 안에 뭐가 들어가나)

각 파라미터 키(예: `model.0.conv.weight`)는 C 식별자로 변환되어:

- `static const float model_0_conv_weight[...] = { ... };`
- `#define MODEL_0_CONV_WEIGHT_ELEMS (...)`
- `#define MODEL_0_CONV_WEIGHT_DIM0 (...)` 같은 shape 매크로

로 출력됩니다.

## 5) MicroBlaze/Vitis에서 쓰는 팁

- `weights.h`를 Vitis 프로젝트 소스에 추가하고 `#include "weights.h"` 하면 됩니다.
- 이 스크립트는 **float32로 강제 변환**해서 내보냅니다. (임베디드에서 다루기 단순)
- 나중에 속도/메모리 최적화가 필요하면:
  - INT8 양자화(quantization)
  - 레이어별로 필요한 weight만 선택 추출
  - 큰 배열을 flash/외부 메모리로 배치
  같은 방향으로 확장하면 됩니다.


# YOLOv5n_in_C
>>>>>>> 83d1ace09113e5e7e9316a7fc2940ce041cca3da
