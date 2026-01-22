#include "bottleneck.h"
#include "conv2d.h"
#include "bn_silu.h"

void bottleneck_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out,
    const float* cv1_gamma, const float* cv1_beta,
    const float* cv1_mean, const float* cv1_var,
    const float* cv2_w, int32_t cv2_c_out,
    const float* cv2_gamma, const float* cv2_beta,
    const float* cv2_mean, const float* cv2_var,
    float eps,
    float* y)
{
    static float cv1_out[1024 * 1024];
    
    // cv1: Conv(1×1) + BN + SiLU
    conv2d_nchw_f32(x, n, c, h, w,
                    cv1_w, cv1_c_out, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv1_out, h, w);
    bn_silu_nchw_f32(cv1_out, n, cv1_c_out, h, w,
                     cv1_gamma, cv1_beta, cv1_mean, cv1_var, eps,
                     cv1_out);
    
    // cv2: Conv(3×3, p=1) + BN + SiLU
    conv2d_nchw_f32(cv1_out, n, cv1_c_out, h, w,
                    cv2_w, cv2_c_out, 3, 3,
                    0, 1, 1, 1, 1, 1,
                    y, h, w);
    bn_silu_nchw_f32(y, n, cv2_c_out, h, w,
                     cv2_gamma, cv2_beta, cv2_mean, cv2_var, eps,
                     y);
    
    // residual connection: y = cv2_out + x
    const int32_t hw = h * w;
    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t ci = 0; ci < cv2_c_out; ci++) {
            const int32_t base = ((ni * cv2_c_out + ci) * h) * w;
            for (int32_t i = 0; i < hw; i++) {
                y[base + i] += x[base + i];
            }
        }
    }
}
