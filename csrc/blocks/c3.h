#ifndef C3_H
#define C3_H

#include <stdint.h>

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
    float* y);

#endif // C3_H
