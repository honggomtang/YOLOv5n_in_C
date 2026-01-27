#include "c3.h"
#include "../operations/conv2d.h"
#include "../operations/silu.h"
#include "../operations/bottleneck.h"
#include "../operations/concat.h"

// Helper: 1x1 fused conv (conv + bias + SiLU)
static void conv1x1(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* w_ptr, int32_t c_out, const float* bias,
    float* y)
{
    conv2d_nchw_f32(x, n, c_in, h, w,
                    w_ptr, c_out, 1, 1,
                    bias, 1, 1, 0, 0, 1,
                    y, h, w);
    silu_nchw_f32(y, n, c_out, h, w, y);
}

void c3_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    const float* cv3_w, int32_t cv3_c_out, const float* cv3_bias,
    int32_t n_bottleneck,
    const float* const* bn_cv1_w, const float* const* bn_cv1_bias,
    const float* const* bn_cv2_w, const float* const* bn_cv2_bias,
    int32_t shortcut,
    float* y)
{
    static float cv1_out[1024 * 1024];
    static float cv2_out[1024 * 1024];
    static float bottleneck_out[1024 * 1024];
    static float bottleneck_tmp[1024 * 1024];
    static float concat_out[2 * 1024 * 1024];
    
    // cv1
    conv1x1(x, n, c_in, h, w, cv1_w, cv1_c_out, cv1_bias, cv1_out);
    
    // cv2 (skip path)
    conv1x1(x, n, c_in, h, w, cv2_w, cv2_c_out, cv2_bias, cv2_out);
    
    // Bottleneck n회
    const float* bn_in = cv1_out;
    float* bn_out = bottleneck_out;
    for (int32_t i = 0; i < n_bottleneck; i++) {
        bn_out = (i % 2 == 0) ? bottleneck_out : bottleneck_tmp;
        
        bottleneck_nchw_f32(
            bn_in, n, cv1_c_out, h, w,
            bn_cv1_w[i], cv1_c_out, bn_cv1_bias[i],
            bn_cv2_w[i], cv1_c_out, bn_cv2_bias[i],
            shortcut,
            bn_out);
        
        bn_in = bn_out;
    }
    
    // Concat
    concat_nchw_f32(bn_out, cv1_c_out, cv2_out, cv2_c_out, n, h, w, concat_out);
    
    // cv3
    conv1x1(concat_out, n, cv1_c_out + cv2_c_out, h, w, cv3_w, cv3_c_out, cv3_bias, y);
}
