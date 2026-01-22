#ifndef CONCAT_H
#define CONCAT_H

#include <stdint.h>

// Channel 차원 기준 concatenation 연산
void concat_nchw_f32(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y);

#endif // CONCAT_H
