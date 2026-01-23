#include <stdio.h>
#include <math.h>

#include "../assets/weights.h"
#include "./test_vectors_layer0_9.h"

#include "../csrc/blocks/conv.h"
#include "../csrc/blocks/c3.h"
#include "../csrc/blocks/sppf.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    // 입력
    const int n = TV_L0_9_X_N;
    const int c_in = TV_L0_9_X_C;
    const int h_in = TV_L0_9_X_H;
    const int w_in = TV_L0_9_X_W;

    // 중간 버퍼들 (각 레이어 출력)
    static float l0_out[TV_L0_N * TV_L0_C * TV_L0_H * TV_L0_W];
    static float l1_out[TV_L1_N * TV_L1_C * TV_L1_H * TV_L1_W];
    static float l2_out[TV_L2_N * TV_L2_C * TV_L2_H * TV_L2_W];
    static float l3_out[TV_L3_N * TV_L3_C * TV_L3_H * TV_L3_W];
    static float l4_out[TV_L4_N * TV_L4_C * TV_L4_H * TV_L4_W];
    static float l5_out[TV_L5_N * TV_L5_C * TV_L5_H * TV_L5_W];
    static float l6_out[TV_L6_N * TV_L6_C * TV_L6_H * TV_L6_W];
    static float l7_out[TV_L7_N * TV_L7_C * TV_L7_H * TV_L7_W];
    static float l8_out[TV_L8_N * TV_L8_C * TV_L8_H * TV_L8_W];
    static float l9_out[TV_L9_N * TV_L9_C * TV_L9_H * TV_L9_W];

    int all_ok = 1;

    // Layer 0: Conv (3->16, k=6, s=2, p=2)
    conv_block_nchw_f32(
        tv_l0_9_x, n, c_in, h_in, w_in,
        model_0_conv_weight, 16, 6, 6,
        2, 2, 2, 2,
        model_0_bn_weight, model_0_bn_bias,
        model_0_bn_running_mean, model_0_bn_running_var,
        1e-3f,
        l0_out, TV_L0_H, TV_L0_W);
    {
        int elems = TV_L0_N * TV_L0_C * TV_L0_H * TV_L0_W;
        float diff = max_abs_diff(l0_out, tv_l0_out, elems);
        printf("Layer 0 diff = %g", diff);
        if (diff < 1e-4f) {
            printf(" OK\n");
        } else {
            printf(" NG\n");
            all_ok = 0;
        }
    }

    // Layer 1: Conv (16->32, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l0_out, n, 16, TV_L0_H, TV_L0_W,
        model_1_conv_weight, 32, 3, 3,
        2, 2, 1, 1,
        model_1_bn_weight, model_1_bn_bias,
        model_1_bn_running_mean, model_1_bn_running_var,
        1e-3f,
        l1_out, TV_L1_H, TV_L1_W);
    {
        int elems = TV_L1_N * TV_L1_C * TV_L1_H * TV_L1_W;
        float diff = max_abs_diff(l1_out, tv_l1_out, elems);
        printf("Layer 1 diff = %g", diff);
        if (diff < 1e-4f) {
            printf(" OK\n");
        } else {
            printf(" NG\n");
            all_ok = 0;
        }
    }

    // Layer 2: C3 (32->32, n_bottleneck=1)
    const float* l2_m_cv1_w[1] = { model_2_m_0_cv1_conv_weight };
    const float* l2_m_cv1_g[1] = { model_2_m_0_cv1_bn_weight };
    const float* l2_m_cv1_b[1] = { model_2_m_0_cv1_bn_bias };
    const float* l2_m_cv1_m[1] = { model_2_m_0_cv1_bn_running_mean };
    const float* l2_m_cv1_v[1] = { model_2_m_0_cv1_bn_running_var };
    const float* l2_m_cv2_w[1] = { model_2_m_0_cv2_conv_weight };
    const float* l2_m_cv2_g[1] = { model_2_m_0_cv2_bn_weight };
    const float* l2_m_cv2_b[1] = { model_2_m_0_cv2_bn_bias };
    const float* l2_m_cv2_m[1] = { model_2_m_0_cv2_bn_running_mean };
    const float* l2_m_cv2_v[1] = { model_2_m_0_cv2_bn_running_var };

    c3_nchw_f32(
        l1_out, n, 32, TV_L1_H, TV_L1_W,
        model_2_cv1_conv_weight, 16,
        model_2_cv1_bn_weight, model_2_cv1_bn_bias,
        model_2_cv1_bn_running_mean, model_2_cv1_bn_running_var,
        model_2_cv2_conv_weight, 16,
        model_2_cv2_bn_weight, model_2_cv2_bn_bias,
        model_2_cv2_bn_running_mean, model_2_cv2_bn_running_var,
        model_2_cv3_conv_weight, 32,
        model_2_cv3_bn_weight, model_2_cv3_bn_bias,
        model_2_cv3_bn_running_mean, model_2_cv3_bn_running_var,
        1,
        l2_m_cv1_w, l2_m_cv1_g, l2_m_cv1_b, l2_m_cv1_m, l2_m_cv1_v,
        l2_m_cv2_w, l2_m_cv2_g, l2_m_cv2_b, l2_m_cv2_m, l2_m_cv2_v,
        1e-3f,
        l2_out);
    {
        int elems = TV_L2_N * TV_L2_C * TV_L2_H * TV_L2_W;
        float diff = max_abs_diff(l2_out, tv_l2_out, elems);
        printf("Layer 2 diff = %g", diff);
        if (diff < 1e-4f) {
            printf(" OK\n");
        } else {
            printf(" NG\n");
            all_ok = 0;
        }
    }

    // Layer 3: Conv (32->64, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l2_out, n, 32, TV_L2_H, TV_L2_W,
        model_3_conv_weight, 64, 3, 3,
        2, 2, 1, 1,
        model_3_bn_weight, model_3_bn_bias,
        model_3_bn_running_mean, model_3_bn_running_var,
        1e-3f,
        l3_out, TV_L3_H, TV_L3_W);
    {
        int elems = TV_L3_N * TV_L3_C * TV_L3_H * TV_L3_W;
        float diff = max_abs_diff(l3_out, tv_l3_out, elems);
        printf("Layer 3 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 4: C3 (64->64, n_bottleneck=2)
    const float* l4_m_cv1_w[2] = { model_4_m_0_cv1_conv_weight, model_4_m_1_cv1_conv_weight };
    const float* l4_m_cv1_g[2] = { model_4_m_0_cv1_bn_weight, model_4_m_1_cv1_bn_weight };
    const float* l4_m_cv1_b[2] = { model_4_m_0_cv1_bn_bias, model_4_m_1_cv1_bn_bias };
    const float* l4_m_cv1_m[2] = { model_4_m_0_cv1_bn_running_mean, model_4_m_1_cv1_bn_running_mean };
    const float* l4_m_cv1_v[2] = { model_4_m_0_cv1_bn_running_var, model_4_m_1_cv1_bn_running_var };
    const float* l4_m_cv2_w[2] = { model_4_m_0_cv2_conv_weight, model_4_m_1_cv2_conv_weight };
    const float* l4_m_cv2_g[2] = { model_4_m_0_cv2_bn_weight, model_4_m_1_cv2_bn_weight };
    const float* l4_m_cv2_b[2] = { model_4_m_0_cv2_bn_bias, model_4_m_1_cv2_bn_bias };
    const float* l4_m_cv2_m[2] = { model_4_m_0_cv2_bn_running_mean, model_4_m_1_cv2_bn_running_mean };
    const float* l4_m_cv2_v[2] = { model_4_m_0_cv2_bn_running_var, model_4_m_1_cv2_bn_running_var };

    c3_nchw_f32(
        l3_out, n, 64, TV_L3_H, TV_L3_W,
        model_4_cv1_conv_weight, 32,
        model_4_cv1_bn_weight, model_4_cv1_bn_bias,
        model_4_cv1_bn_running_mean, model_4_cv1_bn_running_var,
        model_4_cv2_conv_weight, 32,
        model_4_cv2_bn_weight, model_4_cv2_bn_bias,
        model_4_cv2_bn_running_mean, model_4_cv2_bn_running_var,
        model_4_cv3_conv_weight, 64,
        model_4_cv3_bn_weight, model_4_cv3_bn_bias,
        model_4_cv3_bn_running_mean, model_4_cv3_bn_running_var,
        2,
        l4_m_cv1_w, l4_m_cv1_g, l4_m_cv1_b, l4_m_cv1_m, l4_m_cv1_v,
        l4_m_cv2_w, l4_m_cv2_g, l4_m_cv2_b, l4_m_cv2_m, l4_m_cv2_v,
        1e-3f,
        l4_out);
    {
        int elems = TV_L4_N * TV_L4_C * TV_L4_H * TV_L4_W;
        float diff = max_abs_diff(l4_out, tv_l4_out, elems);
        printf("Layer 4 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 5: Conv (64->128, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l4_out, n, 64, TV_L4_H, TV_L4_W,
        model_5_conv_weight, 128, 3, 3,
        2, 2, 1, 1,
        model_5_bn_weight, model_5_bn_bias,
        model_5_bn_running_mean, model_5_bn_running_var,
        1e-3f,
        l5_out, TV_L5_H, TV_L5_W);
    {
        int elems = TV_L5_N * TV_L5_C * TV_L5_H * TV_L5_W;
        float diff = max_abs_diff(l5_out, tv_l5_out, elems);
        printf("Layer 5 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 6: C3 (128->128, n_bottleneck=3)
    const float* l6_m_cv1_w[3] = { model_6_m_0_cv1_conv_weight, model_6_m_1_cv1_conv_weight, model_6_m_2_cv1_conv_weight };
    const float* l6_m_cv1_g[3] = { model_6_m_0_cv1_bn_weight, model_6_m_1_cv1_bn_weight, model_6_m_2_cv1_bn_weight };
    const float* l6_m_cv1_b[3] = { model_6_m_0_cv1_bn_bias, model_6_m_1_cv1_bn_bias, model_6_m_2_cv1_bn_bias };
    const float* l6_m_cv1_m[3] = { model_6_m_0_cv1_bn_running_mean, model_6_m_1_cv1_bn_running_mean, model_6_m_2_cv1_bn_running_mean };
    const float* l6_m_cv1_v[3] = { model_6_m_0_cv1_bn_running_var, model_6_m_1_cv1_bn_running_var, model_6_m_2_cv1_bn_running_var };
    const float* l6_m_cv2_w[3] = { model_6_m_0_cv2_conv_weight, model_6_m_1_cv2_conv_weight, model_6_m_2_cv2_conv_weight };
    const float* l6_m_cv2_g[3] = { model_6_m_0_cv2_bn_weight, model_6_m_1_cv2_bn_weight, model_6_m_2_cv2_bn_weight };
    const float* l6_m_cv2_b[3] = { model_6_m_0_cv2_bn_bias, model_6_m_1_cv2_bn_bias, model_6_m_2_cv2_bn_bias };
    const float* l6_m_cv2_m[3] = { model_6_m_0_cv2_bn_running_mean, model_6_m_1_cv2_bn_running_mean, model_6_m_2_cv2_bn_running_mean };
    const float* l6_m_cv2_v[3] = { model_6_m_0_cv2_bn_running_var, model_6_m_1_cv2_bn_running_var, model_6_m_2_cv2_bn_running_var };

    c3_nchw_f32(
        l5_out, n, 128, TV_L5_H, TV_L5_W,
        model_6_cv1_conv_weight, 64,
        model_6_cv1_bn_weight, model_6_cv1_bn_bias,
        model_6_cv1_bn_running_mean, model_6_cv1_bn_running_var,
        model_6_cv2_conv_weight, 64,
        model_6_cv2_bn_weight, model_6_cv2_bn_bias,
        model_6_cv2_bn_running_mean, model_6_cv2_bn_running_var,
        model_6_cv3_conv_weight, 128,
        model_6_cv3_bn_weight, model_6_cv3_bn_bias,
        model_6_cv3_bn_running_mean, model_6_cv3_bn_running_var,
        3,
        l6_m_cv1_w, l6_m_cv1_g, l6_m_cv1_b, l6_m_cv1_m, l6_m_cv1_v,
        l6_m_cv2_w, l6_m_cv2_g, l6_m_cv2_b, l6_m_cv2_m, l6_m_cv2_v,
        1e-3f,
        l6_out);
    {
        int elems = TV_L6_N * TV_L6_C * TV_L6_H * TV_L6_W;
        float diff = max_abs_diff(l6_out, tv_l6_out, elems);
        printf("Layer 6 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 7: Conv (128->256, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l6_out, n, 128, TV_L6_H, TV_L6_W,
        model_7_conv_weight, 256, 3, 3,
        2, 2, 1, 1,
        model_7_bn_weight, model_7_bn_bias,
        model_7_bn_running_mean, model_7_bn_running_var,
        1e-3f,
        l7_out, TV_L7_H, TV_L7_W);
    {
        int elems = TV_L7_N * TV_L7_C * TV_L7_H * TV_L7_W;
        float diff = max_abs_diff(l7_out, tv_l7_out, elems);
        printf("Layer 7 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 8: C3 (256->256, n_bottleneck=1)
    const float* l8_m_cv1_w[1] = { model_8_m_0_cv1_conv_weight };
    const float* l8_m_cv1_g[1] = { model_8_m_0_cv1_bn_weight };
    const float* l8_m_cv1_b[1] = { model_8_m_0_cv1_bn_bias };
    const float* l8_m_cv1_m[1] = { model_8_m_0_cv1_bn_running_mean };
    const float* l8_m_cv1_v[1] = { model_8_m_0_cv1_bn_running_var };
    const float* l8_m_cv2_w[1] = { model_8_m_0_cv2_conv_weight };
    const float* l8_m_cv2_g[1] = { model_8_m_0_cv2_bn_weight };
    const float* l8_m_cv2_b[1] = { model_8_m_0_cv2_bn_bias };
    const float* l8_m_cv2_m[1] = { model_8_m_0_cv2_bn_running_mean };
    const float* l8_m_cv2_v[1] = { model_8_m_0_cv2_bn_running_var };

    c3_nchw_f32(
        l7_out, n, 256, TV_L7_H, TV_L7_W,
        model_8_cv1_conv_weight, 128,
        model_8_cv1_bn_weight, model_8_cv1_bn_bias,
        model_8_cv1_bn_running_mean, model_8_cv1_bn_running_var,
        model_8_cv2_conv_weight, 128,
        model_8_cv2_bn_weight, model_8_cv2_bn_bias,
        model_8_cv2_bn_running_mean, model_8_cv2_bn_running_var,
        model_8_cv3_conv_weight, 256,
        model_8_cv3_bn_weight, model_8_cv3_bn_bias,
        model_8_cv3_bn_running_mean, model_8_cv3_bn_running_var,
        1,
        l8_m_cv1_w, l8_m_cv1_g, l8_m_cv1_b, l8_m_cv1_m, l8_m_cv1_v,
        l8_m_cv2_w, l8_m_cv2_g, l8_m_cv2_b, l8_m_cv2_m, l8_m_cv2_v,
        1e-3f,
        l8_out);
    {
        int elems = TV_L8_N * TV_L8_C * TV_L8_H * TV_L8_W;
        float diff = max_abs_diff(l8_out, tv_l8_out, elems);
        printf("Layer 8 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 9: SPPF (256->256)
    sppf_nchw_f32(
        l8_out, n, 256, TV_L8_H, TV_L8_W,
        model_9_cv1_conv_weight, 128,
        model_9_cv1_bn_weight, model_9_cv1_bn_bias,
        model_9_cv1_bn_running_mean, model_9_cv1_bn_running_var,
        model_9_cv2_conv_weight, 256,
        model_9_cv2_bn_weight, model_9_cv2_bn_bias,
        model_9_cv2_bn_running_mean, model_9_cv2_bn_running_var,
        5,
        1e-3f,
        l9_out);
    {
        int elems = TV_L9_N * TV_L9_C * TV_L9_H * TV_L9_W;
        float diff = max_abs_diff(l9_out, tv_l9_out, elems);
        printf("Layer 9 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    if (all_ok) {
        printf("\nAll layers OK\n");
        return 0;
    }
    printf("\nSome layers failed\n");
    return 1;
}
