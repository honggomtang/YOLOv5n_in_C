#include <stdio.h>
#include <math.h>

#include "../assets/weights.h"
#include "../tests/test_vectors_c3.h"

#include "blocks/c3.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    // YOLOv5n Layer 2 (C3): 입력 (1, 32, 32, 32) → 출력 (1, 32, 32, 32)
    const int n = TV_C3_X_N;
    const int c_in = TV_C3_X_C;
    const int h = TV_C3_X_H;
    const int w = TV_C3_X_W;
    
    const int c_out = TV_C3_Y_C;
    const int h_out = TV_C3_Y_H;
    const int w_out = TV_C3_Y_W;
    
    static float y_out[1 * 32 * 32 * 32];
    
    // 디버그: 입력 통계
    float x_mean = 0.0f, x_std = 0.0f;
    for (int i = 0; i < n * c_in * h * w; i++) {
        x_mean += tv_c3_x[i];
    }
    x_mean /= (n * c_in * h * w);
    for (int i = 0; i < n * c_in * h * w; i++) {
        float d = tv_c3_x[i] - x_mean;
        x_std += d * d;
    }
    x_std = sqrtf(x_std / (n * c_in * h * w));
    printf("입력 통계: mean=%.6f, std=%.6f\n", x_mean, x_std);
    
    // C3 블록 실행
    c3_nchw_f32(
        tv_c3_x, n, c_in, h, w,
        // cv1: Conv(32→16, 1×1)
        model_2_cv1_conv_weight, 16,
        model_2_cv1_bn_weight, model_2_cv1_bn_bias,
        model_2_cv1_bn_running_mean, model_2_cv1_bn_running_var,
        // cv2: Conv(32→16, 1×1) - skip
        model_2_cv2_conv_weight, 16,
        model_2_cv2_bn_weight, model_2_cv2_bn_bias,
        model_2_cv2_bn_running_mean, model_2_cv2_bn_running_var,
        // cv3: Conv(32→32, 1×1)
        model_2_cv3_conv_weight, 32,
        model_2_cv3_bn_weight, model_2_cv3_bn_bias,
        model_2_cv3_bn_running_mean, model_2_cv3_bn_running_var,
        // bottleneck: n=1
        1,
        // bottleneck.cv1: Conv(16→16, 1×1)
        model_2_m_0_cv1_conv_weight,
        model_2_m_0_cv1_bn_weight, model_2_m_0_cv1_bn_bias,
        model_2_m_0_cv1_bn_running_mean, model_2_m_0_cv1_bn_running_var,
        // bottleneck.cv2: Conv(16→16, 3×3)
        model_2_m_0_cv2_conv_weight,
        model_2_m_0_cv2_bn_weight, model_2_m_0_cv2_bn_bias,
        model_2_m_0_cv2_bn_running_mean, model_2_m_0_cv2_bn_running_var,
        1e-3f,  // eps
        y_out);
    
    // 디버그: 출력 통계
    float y_mean = 0.0f, y_std = 0.0f;
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        y_mean += y_out[i];
    }
    y_mean /= (n * c_out * h_out * w_out);
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        float d = y_out[i] - y_mean;
        y_std += d * d;
    }
    y_std = sqrtf(y_std / (n * c_out * h_out * w_out));
    printf("C 출력 통계: mean=%.6f, std=%.6f\n", y_mean, y_std);
    
    // 파이썬 정답 통계
    float py_mean = 0.0f, py_std = 0.0f;
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        py_mean += tv_c3_y[i];
    }
    py_mean /= (n * c_out * h_out * w_out);
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        float d = tv_c3_y[i] - py_mean;
        py_std += d * d;
    }
    py_std = sqrtf(py_std / (n * c_out * h_out * w_out));
    printf("Python 정답 통계: mean=%.6f, std=%.6f\n", py_mean, py_std);
    
    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_c3_y, elems);
    printf("c3 max_abs_diff = %g\n", diff);
    
    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}
