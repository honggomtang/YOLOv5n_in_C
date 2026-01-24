#ifndef DETECT_H
#define DETECT_H

#include <stdint.h>

/**
 * Classic YOLOv5n Detect Head (Anchor-based)
 *
 * 각 스케일마다 1x1 Conv 하나: P3 64→255, P4 128→255, P5 256→255.
 * 가중치: model.24.m.0, m.1, m.2 (weight, bias).
 * 출력 (1, 255, H, W). 255 = 3 * 85 (x,y,w,h,obj, cls0..79).
 */

void detect_nchw_f32(
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const float* m0_w, const float* m0_b,
    const float* m1_w, const float* m1_b,
    const float* m2_w, const float* m2_b,
    float* p3_out, float* p4_out, float* p5_out);

#endif /* DETECT_H */
