#include <stdio.h>
#include <math.h>

#include "../assets/weights.h"
#include "../tests/test_vectors_c3_intermediate.h"

#include "blocks/c3.h"

static float compute_mean(const float* a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i];
    return sum / n;
}

static float compute_std(const float* a, int n, float mean) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - mean;
        sum += d * d;
    }
    return sqrtf(sum / n);
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int n = TV_C3_X_N;
    const int c_in = TV_C3_X_C;
    const int h = TV_C3_X_H;
    const int w = TV_C3_X_W;
    
    static float cv1_out[1 * 16 * 32 * 32];
    static float bottleneck_out[1 * 16 * 32 * 32];
    static float cv2_out[1 * 16 * 32 * 32];
    static float concat_out[1 * 32 * 32 * 32];
    static float cv3_out[1 * 32 * 32 * 32];
    
    // 단계별 실행 및 비교
    
    // 1. cv1
    #include "operations/conv2d.h"
    #include "operations/bn_silu.h"
    
    conv2d_nchw_f32(tv_c3_x, n, c_in, h, w,
                    model_2_cv1_conv_weight, 16, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv1_out, h, w);
    bn_silu_nchw_f32(cv1_out, n, 16, h, w,
                     model_2_cv1_bn_weight, model_2_cv1_bn_bias,
                     model_2_cv1_bn_running_mean, model_2_cv1_bn_running_var,
                     1e-3f, cv1_out);
    
    int cv1_size = n * 16 * h * w;
    float cv1_mean = compute_mean(cv1_out, cv1_size);
    float cv1_std = compute_std(cv1_out, cv1_size, cv1_mean);
    float cv1_diff = max_abs_diff(cv1_out, tv_c3_cv1_out, cv1_size);
    printf("cv1: C mean=%.6f std=%.6f, Py mean=%.6f std=%.6f, diff=%.6f\n",
           cv1_mean, cv1_std,
           compute_mean(tv_c3_cv1_out, cv1_size),
           compute_std(tv_c3_cv1_out, cv1_size, compute_mean(tv_c3_cv1_out, cv1_size)),
           cv1_diff);
    
    // 2. bottleneck
    #include "operations/bottleneck.h"
    bottleneck_nchw_f32(cv1_out, n, 16, h, w,
                        model_2_m_0_cv1_conv_weight, 16,
                        model_2_m_0_cv1_bn_weight, model_2_m_0_cv1_bn_bias,
                        model_2_m_0_cv1_bn_running_mean, model_2_m_0_cv1_bn_running_var,
                        model_2_m_0_cv2_conv_weight, 16,
                        model_2_m_0_cv2_bn_weight, model_2_m_0_cv2_bn_bias,
                        model_2_m_0_cv2_bn_running_mean, model_2_m_0_cv2_bn_running_var,
                        1e-3f, bottleneck_out);
    
    int bottleneck_size = n * 16 * h * w;
    float bottleneck_mean = compute_mean(bottleneck_out, bottleneck_size);
    float bottleneck_std = compute_std(bottleneck_out, bottleneck_size, bottleneck_mean);
    float bottleneck_diff = max_abs_diff(bottleneck_out, tv_c3_bottleneck_out, bottleneck_size);
    printf("bottleneck: C mean=%.6f std=%.6f, Py mean=%.6f std=%.6f, diff=%.6f\n",
           bottleneck_mean, bottleneck_std,
           compute_mean(tv_c3_bottleneck_out, bottleneck_size),
           compute_std(tv_c3_bottleneck_out, bottleneck_size, compute_mean(tv_c3_bottleneck_out, bottleneck_size)),
           bottleneck_diff);
    
    // 3. cv2
    conv2d_nchw_f32(tv_c3_x, n, c_in, h, w,
                    model_2_cv2_conv_weight, 16, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv2_out, h, w);
    bn_silu_nchw_f32(cv2_out, n, 16, h, w,
                     model_2_cv2_bn_weight, model_2_cv2_bn_bias,
                     model_2_cv2_bn_running_mean, model_2_cv2_bn_running_var,
                     1e-3f, cv2_out);
    
    int cv2_size = n * 16 * h * w;
    float cv2_mean = compute_mean(cv2_out, cv2_size);
    float cv2_std = compute_std(cv2_out, cv2_size, cv2_mean);
    float cv2_diff = max_abs_diff(cv2_out, tv_c3_cv2_out, cv2_size);
    printf("cv2: C mean=%.6f std=%.6f, Py mean=%.6f std=%.6f, diff=%.6f\n",
           cv2_mean, cv2_std,
           compute_mean(tv_c3_cv2_out, cv2_size),
           compute_std(tv_c3_cv2_out, cv2_size, compute_mean(tv_c3_cv2_out, cv2_size)),
           cv2_diff);
    
    // 4. concat
    #include "operations/concat.h"
    concat_nchw_f32(bottleneck_out, 16,
                    cv2_out, 16,
                    n, h, w,
                    concat_out);
    
    int concat_size = n * 32 * h * w;
    float concat_mean = compute_mean(concat_out, concat_size);
    float concat_std = compute_std(concat_out, concat_size, concat_mean);
    float concat_diff = max_abs_diff(concat_out, tv_c3_concat_out, concat_size);
    printf("concat: C mean=%.6f std=%.6f, Py mean=%.6f std=%.6f, diff=%.6f\n",
           concat_mean, concat_std,
           compute_mean(tv_c3_concat_out, concat_size),
           compute_std(tv_c3_concat_out, concat_size, compute_mean(tv_c3_concat_out, concat_size)),
           concat_diff);
    
    // 5. cv3
    conv2d_nchw_f32(concat_out, n, 32, h, w,
                    model_2_cv3_conv_weight, 32, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    cv3_out, h, w);
    bn_silu_nchw_f32(cv3_out, n, 32, h, w,
                     model_2_cv3_bn_weight, model_2_cv3_bn_bias,
                     model_2_cv3_bn_running_mean, model_2_cv3_bn_running_var,
                     1e-3f, cv3_out);
    
    int cv3_size = n * 32 * h * w;
    float cv3_mean = compute_mean(cv3_out, cv3_size);
    float cv3_std = compute_std(cv3_out, cv3_size, cv3_mean);
    float cv3_diff = max_abs_diff(cv3_out, tv_c3_cv3_out, cv3_size);
    printf("cv3: C mean=%.6f std=%.6f, Py mean=%.6f std=%.6f, diff=%.6f\n",
           cv3_mean, cv3_std,
           compute_mean(tv_c3_cv3_out, cv3_size),
           compute_std(tv_c3_cv3_out, cv3_size, compute_mean(tv_c3_cv3_out, cv3_size)),
           cv3_diff);
    
    return 0;
}
