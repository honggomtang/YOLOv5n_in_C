#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include <stdint.h>

// Nearest neighbor upsampling (NCHW)
// scale_factor=2: 각 픽셀을 2×2로 복제
void upsample_nearest2x_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

#endif // UPSAMPLE_H
