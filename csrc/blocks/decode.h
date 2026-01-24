#ifndef DECODE_H
#define DECODE_H

#include <stdint.h>

// Detection 결과 구조체 (내부 처리용, float)
typedef struct {
    float x, y, w, h;   // 중심 좌표 및 크기 (normalized)
    float conf;         // confidence score
    int32_t cls_id;     // class ID
} detection_t;

// HW 출력용 구조체 (8 bytes, 고정 크기)
// 실제 FPGA에서 호스트로 전송하는 형식
typedef struct __attribute__((packed)) {
    uint16_t x, y, w, h;   // 픽셀 좌표 (정수, 0~65535)
    uint8_t  class_id;     // 클래스 ID (0~79)
    uint8_t  confidence;   // 신뢰도 (0~255, conf*255)
    uint8_t  reserved[2];  // 8바이트 정렬용
} hw_detection_t;

/**
 * Classic YOLOv5n Anchor-based Decode
 *
 * 입력: (1, 255, H, W) x 3 scale. 255 = 3 * 85 (x,y,w,h,obj, cls0..79).
 * Layout: [anchor0_85, anchor1_85, anchor2_85] (channel-major).
 *
 * conf = obj_conf * max_cls_conf.
 * xy = (sigmoid(xy)*2 + grid) * stride, grid = (x,y) - 0.5.
 * wh = (sigmoid(wh)*2)^2 * anchor (pixel).
 */

int32_t decode_nchw_f32(
    const float* p3, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_h, int32_t p5_w,
    int32_t num_classes,
    float conf_threshold,
    int32_t input_size,
    const float strides[3],
    const float anchors[3][6],  /* P3/P4/P5 each [aw0,ah0, aw1,ah1, aw2,ah2] */
    detection_t* detections,
    int32_t max_detections);

#endif /* DECODE_H */
