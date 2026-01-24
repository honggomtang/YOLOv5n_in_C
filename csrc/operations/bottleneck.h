#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <stdint.h>

// Fused Bottleneck: cv1(1x1) + cv2(3x3) + optional shortcut
void bottleneck_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    int32_t shortcut,  // 1=add residual, 0=no shortcut
    float* y);

#endif // BOTTLENECK_H
