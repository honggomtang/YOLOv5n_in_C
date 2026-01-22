#include <stdio.h>
#include <math.h>

#include "../assets/weights.h"
#include "../tests/test_vectors_conv0.h"

#include "operations/conv2d.h"
#include "operations/bn_silu.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    // YOLOv5n conv0: Conv2d(3->16, k=6, s=2, p=2), bias 없음
    const int n = TV_X_N;
    const int c_in = TV_X_C;
    const int h_in = TV_X_H;
    const int w_in = TV_X_W;

    const int c_out = 16;
    const int k = 6;
    const int stride = 2;
    const int pad = 2;
    const int h_out = TV_Y_H;
    const int w_out = TV_Y_W;

    static float y_conv[1 * 16 * TV_Y_H * TV_Y_W];
    static float y_out[1 * 16 * TV_Y_H * TV_Y_W];

    conv2d_nchw_f32(
        tv_x, n, c_in, h_in, w_in,
        model_0_conv_weight, c_out, k, k,
        0,
        stride, stride,
        pad, pad,
        1,
        y_conv, h_out, w_out);

    bn_silu_nchw_f32(
        y_conv, n, c_out, h_out, w_out,
        model_0_bn_weight, model_0_bn_bias,
        model_0_bn_running_mean, model_0_bn_running_var,
        1e-3f,
        y_out);

    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_y, elems);
    printf("conv0 max_abs_diff = %g\n", diff);

    // 대충 이 정도면 맞는 걸로 보자
    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}

