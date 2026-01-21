import torch
from ultralytics import YOLO

# 1. 모델 불러오기 (ultralytics가 내부적으로 모델 구조를 알아서 잡아줍니다)
print("Loading YOLOv5n model...")
model = YOLO('yolov5n.pt')

# 2. 가중치(Weights) 데이터 추출
# 가중치만 담고 있는 state_dict를 가져옵니다.
weights = model.model.state_dict()

# 3. C 헤더 파일 생성
with open('weights.h', 'w') as f:
    f.write("#ifndef WEIGHTS_H\n")
    f.write("#define WEIGHTS_H\n\n")
    
    print("Extracting weights to weights.h...")
    
    for name, param in weights.items():
        # 점(.)이나 특수문자를 언더바(_)로 바꿔서 C 변수명으로 만듭니다.
        clean_name = name.replace('.', '_')
        data = param.cpu().numpy().flatten()
        
        # 배열 선언 (float 형식)
        f.write(f"// Layer: {name}\n")
        f.write(f"static const float {clean_name}[] = {{\n    ")
        
        # 숫자들을 10개씩 끊어서 기록 (파일이 너무 길어지는 것 방지)
        for i, val in enumerate(data):
            f.write(f"{val:.8f}f, ")
            if (i + 1) % 10 == 0:
                f.write("\n    ")
        
        f.write("\n};\n\n")
    
    f.write("#endif // WEIGHTS_H\n")

print("Successfully created weights.h!")
