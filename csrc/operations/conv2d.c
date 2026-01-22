#include "conv2d.h"

void conv2d_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out)
{
    if (groups != 1) {
        return;
    }

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t oc = 0; oc < c_out; oc++) {
            for (int32_t oh = 0; oh < h_out; oh++) {
                for (int32_t ow = 0; ow < w_out; ow++) {
                    float acc = bias_or_null ? bias_or_null[oc] : 0.0f;
                    for (int32_t ic = 0; ic < c_in; ic++) {
                        for (int32_t kh = 0; kh < k_h; kh++) {
                            for (int32_t kw = 0; kw < k_w; kw++) {
                                const int32_t ih = oh * stride_h - pad_h + kh;
                                const int32_t iw = ow * stride_w - pad_w + kw;
                                if ((uint32_t)ih >= (uint32_t)h_in || (uint32_t)iw >= (uint32_t)w_in) {
                                    continue;
                                }

                                const int32_t x_idx = ((ni * c_in + ic) * h_in + ih) * w_in + iw;
                                const int32_t w_idx = (((oc * c_in + ic) * k_h) + kh) * k_w + kw;
                                acc += x[x_idx] * w[w_idx];
                            }
                        }
                    }
                    const int32_t y_idx = ((ni * c_out + oc) * h_out + oh) * w_out + ow;
                    y[y_idx] = acc;
                }
            }
        }
    }
}
