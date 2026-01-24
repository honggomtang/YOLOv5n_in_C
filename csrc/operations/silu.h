#ifndef SILU_H
#define SILU_H

#include <stdint.h>

// SiLU 활성화 함수 (Fused Conv용)
void silu_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

#endif // SILU_H
