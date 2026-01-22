#include "c3.h"
#include "../operations/conv2d.h"
#include "../operations/bn_silu.h"
#include "../operations/bottleneck.h"
#include "../operations/concat.h"

// C3 블록: Cross-stage partial bottleneck
void c3_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out,
    const float* cv1_gamma, const float* cv1_beta,
    const float* cv1_mean, const float* cv1_var,
    const float* cv2_w, int32_t cv2_c_out,
    const float* cv2_gamma, const float* cv2_beta,
    const float* cv2_mean, const float* cv2_var,
    const float* cv3_w, int32_t cv3_c_out,
    const float* cv3_gamma, const float* cv3_beta,
    const float* cv3_mean, const float* cv3_var,
    int32_t n_bottleneck,
    const float* bottleneck_cv1_w, const float* bottleneck_cv1_gamma,
    const float* bottleneck_cv1_beta, const float* bottleneck_cv1_mean,
    const float* bottleneck_cv1_var,
    const float* bottleneck_cv2_w, const float* bottleneck_cv2_gamma,
    const float* bottleneck_cv2_beta, const float* bottleneck_cv2_mean,
    const float* bottleneck_cv2_var,
    float eps,
    float* y)
{
    static float cv1_out[1024 * 1024];
    static float cv2_out[1024 * 1024];
    static float bottleneck_out[1024 * 1024];
    static float concat_out[1024 * 1024];
    
    // cv1 경로: x → cv1 → bottleneck n회
    conv2d_nchw_f32(x, n, c_in, h, w,
                    cv1_w, cv1_c_out, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv1_out, h, w);
    bn_silu_nchw_f32(cv1_out, n, cv1_c_out, h, w,
                     cv1_gamma, cv1_beta, cv1_mean, cv1_var, eps,
                     cv1_out);
    
    // bottleneck n회 반복
    for (int32_t i = 0; i < n_bottleneck; i++) {
        // TODO: bottleneck 파라미터 접근 방식 수정 필요
        bottleneck_nchw_f32(
            (i == 0) ? cv1_out : bottleneck_out,
            n, cv1_c_out, h, w,
            bottleneck_cv1_w, cv1_c_out,
            bottleneck_cv1_gamma, bottleneck_cv1_beta,
            bottleneck_cv1_mean, bottleneck_cv1_var,
            bottleneck_cv2_w, cv1_c_out,
            bottleneck_cv2_gamma, bottleneck_cv2_beta,
            bottleneck_cv2_mean, bottleneck_cv2_var,
            eps,
            bottleneck_out);
    }
    
    // cv2 경로 (skip): x → cv2
    conv2d_nchw_f32(x, n, c_in, h, w,
                    cv2_w, cv2_c_out, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv2_out, h, w);
    bn_silu_nchw_f32(cv2_out, n, cv2_c_out, h, w,
                     cv2_gamma, cv2_beta, cv2_mean, cv2_var, eps,
                     cv2_out);
    
    // concat([bottleneck_out, cv2_out])
    // PyTorch: torch.cat([bottleneck_out, cv2_out], 1)
    // 순서: bottleneck_out 먼저, cv2_out 나중
    concat_nchw_f32(bottleneck_out, cv1_c_out,  // x1: bottleneck_out (cv1_c_out 채널)
                    cv2_out, cv2_c_out,         // x2: cv2_out (cv2_c_out 채널)
                    n, h, w,
                    concat_out);
    
    // cv3: Conv(1×1) + BN + SiLU
    conv2d_nchw_f32(concat_out, n, cv1_c_out + cv2_c_out, h, w,
                    cv3_w, cv3_c_out, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    y, h, w);
    bn_silu_nchw_f32(y, n, cv3_c_out, h, w,
                     cv3_gamma, cv3_beta, cv3_mean, cv3_var, eps,
                     y);
}
