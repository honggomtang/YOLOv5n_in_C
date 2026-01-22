#include "conv.h"
#include "../operations/conv2d.h"
#include "../operations/bn_silu.h"

// Conv 블록: Conv2D + BN + SiLU
void conv_block_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* gamma, const float* beta,
    const float* mean, const float* var,
    float eps,
    float* y, int32_t h_out, int32_t w_out)
{
    static float conv_out[1024 * 1024];
    
    // Conv2D
    conv2d_nchw_f32(x, n, c_in, h_in, w_in,
                    w, c_out, k_h, k_w,
                    0,  // bias 없음
                    stride_h, stride_w,
                    pad_h, pad_w,
                    1,  // groups
                    conv_out, h_out, w_out);
    
    // BN + SiLU
    bn_silu_nchw_f32(conv_out, n, c_out, h_out, w_out,
                     gamma, beta, mean, var, eps,
                     y);
}
