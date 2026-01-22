#include "concat.h"

void concat_nchw_f32(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y)
{
    const int32_t hw = h * w;
    
    for (int32_t ni = 0; ni < n; ni++) {
        // x1 채널들 복사
        for (int32_t ci = 0; ci < c1; ci++) {
            const int32_t src_base = ((ni * c1 + ci) * h) * w;
            const int32_t dst_base = ((ni * (c1 + c2) + ci) * h) * w;
            for (int32_t i = 0; i < hw; i++) {
                y[dst_base + i] = x1[src_base + i];
            }
        }
        // x2 채널들 복사
        for (int32_t ci = 0; ci < c2; ci++) {
            const int32_t src_base = ((ni * c2 + ci) * h) * w;
            const int32_t dst_base = ((ni * (c1 + c2) + (c1 + ci)) * h) * w;
            for (int32_t i = 0; i < hw; i++) {
                y[dst_base + i] = x2[src_base + i];
            }
        }
    }
}
