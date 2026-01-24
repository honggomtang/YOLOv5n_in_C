#ifndef CONV_H
#define CONV_H

#include <stdint.h>

// Fused Conv 블록: Conv2D + Bias + SiLU
void conv_block_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* bias,  // Fused bias (BN 흡수됨)
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV_H
