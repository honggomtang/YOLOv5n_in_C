#ifndef C3_H
#define C3_H

#include <stdint.h>

// Fused C3 블록
void c3_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    const float* cv3_w, int32_t cv3_c_out, const float* cv3_bias,
    int32_t n_bottleneck,
    const float* const* bn_cv1_w, const float* const* bn_cv1_bias,
    const float* const* bn_cv2_w, const float* const* bn_cv2_bias,
    int32_t shortcut,  // 1=add residual in bottleneck, 0=no shortcut
    float* y);

#endif // C3_H
