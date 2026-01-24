#include "sppf.h"
#include "../operations/conv2d.h"
#include "../operations/silu.h"
#include "../operations/maxpool2d.h"
#include "../operations/concat.h"

void sppf_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    int32_t pool_k,
    float* y)
{
    const int32_t pad = pool_k / 2;

    static float x1[1024 * 1024];
    static float y1[1024 * 1024];
    static float y2[1024 * 1024];
    static float y3[1024 * 1024];
    static float cat[1024 * 1024];
    static float cv2_out[1024 * 1024];

    // cv1: 1x1 Conv + Bias + SiLU
    conv2d_nchw_f32(x, n, c_in, h, w,
                    cv1_w, cv1_c_out, 1, 1,
                    cv1_bias, 1, 1, 0, 0, 1,
                    x1, h, w);
    silu_nchw_f32(x1, n, cv1_c_out, h, w, x1);

    // MaxPool 3회
    maxpool2d_nchw_f32(x1, n, cv1_c_out, h, w, pool_k, 1, pad, y1, h, w);
    maxpool2d_nchw_f32(y1, n, cv1_c_out, h, w, pool_k, 1, pad, y2, h, w);
    maxpool2d_nchw_f32(y2, n, cv1_c_out, h, w, pool_k, 1, pad, y3, h, w);

    // Concat [x1, y1, y2, y3]
    concat4_nchw_f32(x1, cv1_c_out, y1, cv1_c_out, y2, cv1_c_out, y3, cv1_c_out,
                     n, h, w, cat);

    // cv2: 1x1 Conv + Bias + SiLU
    conv2d_nchw_f32(cat, n, 4 * cv1_c_out, h, w,
                    cv2_w, cv2_c_out, 1, 1,
                    cv2_bias, 1, 1, 0, 0, 1,
                    cv2_out, h, w);
    silu_nchw_f32(cv2_out, n, cv2_c_out, h, w, y);
}
