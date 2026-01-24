#ifndef SPPF_H
#define SPPF_H

#include <stdint.h>

// Fused SPPF 블록
void sppf_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    int32_t pool_k,
    float* y);

#endif // SPPF_H
