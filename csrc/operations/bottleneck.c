#include "bottleneck.h"
#include "conv2d.h"
#include "silu.h"

void bottleneck_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out, const float* cv1_bias,
    const float* cv2_w, int32_t cv2_c_out, const float* cv2_bias,
    int32_t shortcut,
    float* y)
{
    static float cv1_out[1024 * 1024];
    static float cv2_out[1024 * 1024];
    
    // cv1: 1x1 Conv + Bias + SiLU
    conv2d_nchw_f32(x, n, c, h, w,
                    cv1_w, cv1_c_out, 1, 1,
                    cv1_bias, 1, 1, 0, 0, 1,
                    cv1_out, h, w);
    silu_nchw_f32(cv1_out, n, cv1_c_out, h, w, cv1_out);
    
    // cv2: 3x3 Conv + Bias + SiLU
    conv2d_nchw_f32(cv1_out, n, cv1_c_out, h, w,
                    cv2_w, cv2_c_out, 3, 3,
                    cv2_bias, 1, 1, 1, 1, 1,
                    cv2_out, h, w);
    silu_nchw_f32(cv2_out, n, cv2_c_out, h, w, cv2_out);
    
    // Shortcut (only if shortcut=1 and dimensions match)
    if (shortcut && c == cv2_c_out) {
        int32_t size = n * c * h * w;
        for (int32_t i = 0; i < size; i++) {
            y[i] = x[i] + cv2_out[i];
        }
    } else {
        int32_t size = n * cv2_c_out * h * w;
        for (int32_t i = 0; i < size; i++) {
            y[i] = cv2_out[i];
        }
    }
}
