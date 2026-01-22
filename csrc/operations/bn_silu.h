#ifndef BN_SILU_H
#define BN_SILU_H

#include <stdint.h>

// BatchNorm + SiLU activation 연산
void bn_silu_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* gamma, const float* beta,
    const float* mean, const float* var,
    float eps,
    float* y);

#endif // BN_SILU_H
