/**
 * Classic YOLOv5n Detect Head (Anchor-based)
 *
 * P3(64,H,W) -> Conv 1x1 64->255 -> (255,H,W)
 * P4(128,H,W) -> Conv 1x1 128->255 -> (255,H,W)
 * P5(256,H,W) -> Conv 1x1 256->255 -> (255,H,W)
 *
 * 가중치: model.24.m.0, m.1, m.2 (weight, bias).
 */

#include "detect.h"
#include "../operations/conv2d.h"

void detect_nchw_f32(
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const float* m0_w, const float* m0_b,
    const float* m1_w, const float* m1_b,
    const float* m2_w, const float* m2_b,
    float* p3_out, float* p4_out, float* p5_out)
{
    conv2d_nchw_f32(p3, 1, p3_c, p3_h, p3_w,
        m0_w, 255, 1, 1, m0_b, 1, 1, 0, 0, 1,
        p3_out, p3_h, p3_w);
    conv2d_nchw_f32(p4, 1, p4_c, p4_h, p4_w,
        m1_w, 255, 1, 1, m1_b, 1, 1, 0, 0, 1,
        p4_out, p4_h, p4_w);
    conv2d_nchw_f32(p5, 1, p5_c, p5_h, p5_w,
        m2_w, 255, 1, 1, m2_b, 1, 1, 0, 0, 1,
        p5_out, p5_h, p5_w);
}
