#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <stdint.h>

// Bottleneck 연산: cv1(1×1) → cv2(3×3)
void bottleneck_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out,
    const float* cv1_gamma, const float* cv1_beta,
    const float* cv1_mean, const float* cv1_var,
    const float* cv2_w, int32_t cv2_c_out,
    const float* cv2_gamma, const float* cv2_beta,
    const float* cv2_mean, const float* cv2_var,
    float eps,
    float* y);

#endif // BOTTLENECK_H
