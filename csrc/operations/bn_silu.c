#include "bn_silu.h"

static float silu_f32(float x) {
    extern float expf(float);
    const float s = 1.0f / (1.0f + expf(-x));
    return x * s;
}

void bn_silu_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* gamma, const float* beta,
    const float* mean, const float* var,
    float eps,
    float* y)
{
    const int32_t hw = h * w;
    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t ci = 0; ci < c; ci++) {
            const float g = gamma[ci];
            const float b = beta[ci];
            const float m = mean[ci];
            const float v = var[ci];
            extern float sqrtf(float);
            const float inv_std = 1.0f / sqrtf(v + eps);

            const int32_t base = (ni * c + ci) * hw;
            for (int32_t i = 0; i < hw; i++) {
                float z = (x[base + i] - m) * inv_std;
                z = z * g + b;
                y[base + i] = silu_f32(z);
            }
        }
    }
}
