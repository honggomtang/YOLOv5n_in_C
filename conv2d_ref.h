#ifndef CONV2D_REF_H
#define CONV2D_REF_H

#include <stdint.h>

// NCHW 기준 conv2d 레퍼런스 구현
// weight는 OIHW(=out,in,kh,kw)로 평탄화돼있다고 가정
void conv2d_nchw_ref_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV2D_REF_H

