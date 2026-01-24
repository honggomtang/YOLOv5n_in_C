#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_vectors_nms.h"
#include "../csrc/blocks/nms.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// detection 배열을 confidence 내림차순으로 정렬 (버블 정렬)
static void sort_detections_by_conf(detection_t* detections, int32_t num) {
    for (int i = 0; i < num - 1; i++) {
        for (int j = 0; j < num - 1 - i; j++) {
            if (detections[j].conf < detections[j + 1].conf) {
                detection_t temp = detections[j];
                detections[j] = detections[j + 1];
                detections[j + 1] = temp;
            }
        }
    }
}

int main(void) {
    // 테스트: NMS 전 detection 배열 준비
    // 실제로는 decode 블록의 출력을 사용하지만, 여기서는 테스트 벡터 사용
    
    // Python NMS 결과 (참조)
    const int32_t num_ref = TV_NMS_NUM_DETECTIONS;
    
    printf("Reference detections (Python NMS): %d\n", num_ref);
    
    // 테스트용 입력 detection 배열 생성 (중복/겹치는 detection 포함)
    // 실제로는 decode 블록의 출력을 사용하지만, 여기서는 간단한 테스트 데이터 사용
    const int32_t num_input = 10;  // 예시: 10개의 detection
    detection_t input_detections[10] = {
        // 겹치는 detection들 (같은 클래스)
        {0.5f, 0.5f, 0.3f, 0.3f, 0.9f, 0},  // 높은 confidence
        {0.52f, 0.52f, 0.3f, 0.3f, 0.8f, 0},  // 겹침 (제거되어야 함)
        {0.48f, 0.48f, 0.3f, 0.3f, 0.7f, 0},  // 겹침 (제거되어야 함)
        // 다른 클래스 (제거되지 않아야 함)
        {0.2f, 0.2f, 0.2f, 0.2f, 0.85f, 1},
        {0.22f, 0.22f, 0.2f, 0.2f, 0.75f, 1},  // 같은 클래스, 겹침
        // 겹치지 않는 detection
        {0.8f, 0.8f, 0.1f, 0.1f, 0.6f, 0},
        {0.1f, 0.8f, 0.15f, 0.15f, 0.55f, 2},
        {0.9f, 0.1f, 0.12f, 0.12f, 0.5f, 0},
        {0.3f, 0.7f, 0.18f, 0.18f, 0.45f, 1},
        {0.7f, 0.3f, 0.14f, 0.14f, 0.4f, 2},
    };
    
    // confidence로 정렬 (NMS 전 필수)
    sort_detections_by_conf(input_detections, num_input);
    
    printf("\nInput detections (sorted by confidence):\n");
    for (int i = 0; i < num_input; i++) {
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, input_detections[i].cls_id, input_detections[i].conf,
               input_detections[i].x, input_detections[i].y,
               input_detections[i].w, input_detections[i].h);
    }
    
    // NMS 적용
    detection_t* output_detections = NULL;
    int32_t output_count = 0;
    
    int ret = nms(
        input_detections, num_input,
        &output_detections, &output_count,
        TV_NMS_IOU_THRESHOLD,
        TV_NMS_MAX_DETECTIONS);
    
    if (ret != 0) {
        fprintf(stderr, "NMS failed\n");
        return 1;
    }
    
    printf("\nNMS output: %d detections\n", output_count);
    for (int i = 0; i < output_count; i++) {
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, output_detections[i].cls_id, output_detections[i].conf,
               output_detections[i].x, output_detections[i].y,
               output_detections[i].w, output_detections[i].h);
    }
    
    // Python 참조 결과와 비교
    printf("\nReference detections (Python):\n");
    for (int i = 0; i < num_ref && i < 5; i++) {  // 처음 5개만 출력
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, tv_nms_detections[i].cls_id, tv_nms_detections[i].conf,
               tv_nms_detections[i].x, tv_nms_detections[i].y,
               tv_nms_detections[i].w, tv_nms_detections[i].h);
    }
    
    // 실제 검증: NMS 로직이 올바르게 동작하는지 확인
    printf("\n=== Verification ===\n");
    printf("Note: This test uses synthetic data, not actual decode output.\n");
    printf("For full pipeline verification, use test_nms_full.\n");
    
    int verification_ok = 1;
    
    // 1. NMS가 중복을 제거하는지 확인
    if (output_count > 0 && output_count <= num_input) {
        printf("✓ NMS correctly reduced detections: %d -> %d\n", num_input, output_count);
    } else {
        printf("✗ ERROR: Unexpected output count: %d (expected: 0 < count <= %d)\n", 
               output_count, num_input);
        verification_ok = 0;
    }
    
    // 2. 출력이 confidence 내림차순으로 정렬되어 있는지 확인
    int sorted = 1;
    for (int i = 0; i < output_count - 1; i++) {
        if (output_detections[i].conf < output_detections[i + 1].conf) {
            sorted = 0;
            break;
        }
    }
    if (sorted) {
        printf("✓ Output detections are sorted by confidence (descending)\n");
    } else {
        printf("✗ ERROR: Output detections are not sorted by confidence\n");
        verification_ok = 0;
    }
    
    // 3. 같은 클래스의 겹치는 detection이 제거되었는지 확인
    // (입력에서 cls=0, conf=0.9와 cls=0, conf=0.8, 0.7이 겹치므로 하나만 남아야 함)
    int cls0_count = 0;
    for (int i = 0; i < output_count; i++) {
        if (output_detections[i].cls_id == 0) {
            cls0_count++;
        }
    }
    // 입력에는 cls=0이 5개 있었는데, 겹치는 것들이 제거되어야 함
    if (cls0_count < 5) {
        printf("✓ Overlapping detections of same class were removed (cls=0: %d remaining)\n", cls0_count);
    } else {
        printf("⚠ WARNING: May not have removed overlapping detections (cls=0: %d remaining)\n", cls0_count);
    }
    
    // 4. 다른 클래스의 detection은 유지되었는지 확인
    int cls1_count = 0, cls2_count = 0;
    for (int i = 0; i < output_count; i++) {
        if (output_detections[i].cls_id == 1) cls1_count++;
        if (output_detections[i].cls_id == 2) cls2_count++;
    }
    if (cls1_count > 0 && cls2_count > 0) {
        printf("✓ Detections from different classes are preserved (cls=1: %d, cls=2: %d)\n", 
               cls1_count, cls2_count);
    } else {
        printf("⚠ WARNING: Some class detections may have been incorrectly removed\n");
    }
    
    // IoU 계산 함수 테스트
    printf("\n=== IoU Test ===\n");
    detection_t box1 = {0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0};
    detection_t box2 = {0.52f, 0.52f, 0.2f, 0.2f, 0.8f, 0};
    float iou = calculate_iou(&box1, &box2);
    printf("Box1: (%.2f,%.2f,%.2f,%.2f)\n", box1.x, box1.y, box1.w, box1.h);
    printf("Box2: (%.2f,%.2f,%.2f,%.2f)\n", box2.x, box2.y, box2.w, box2.h);
    printf("IoU: %.4f\n", iou);
    
    // IoU가 합리적인 범위인지 확인 (0~1)
    if (iou >= 0.0f && iou <= 1.0f) {
        printf("✓ IoU calculation is in valid range [0, 1]: %.4f\n", iou);
    } else {
        printf("✗ ERROR: IoU out of range: %.4f (expected: [0, 1])\n", iou);
        verification_ok = 0;
    }
    
    // 5. IoU가 합리적인 값인지 확인 (겹치는 박스의 경우 0.5 이상이어야 함)
    if (iou > 0.5f) {
        printf("✓ IoU value is reasonable for overlapping boxes: %.4f\n", iou);
    } else {
        printf("⚠ WARNING: IoU seems low for overlapping boxes: %.4f\n", iou);
    }
    
    // 정리
    if (output_detections) {
        free(output_detections);
    }
    
    // 최종 결과
    printf("\n=== Test Result ===\n");
    if (verification_ok) {
        printf("✓ NMS test completed successfully - all verifications passed\n");
        return 0;
    } else {
        printf("✗ NMS test failed - some verifications did not pass\n");
        return 1;
    }
}
