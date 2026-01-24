#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils/weights_loader.h"
#include "utils/image_loader.h"
#include "blocks/conv.h"
#include "blocks/c3.h"
#include "blocks/sppf.h"
#include "blocks/detect.h"
#include "blocks/decode.h"
#include "blocks/nms.h"
#include "operations/upsample.h"
#include "operations/concat.h"

static int detection_has_nan_bbox(const detection_t* d) {
    return (d->x != d->x) || (d->y != d->y) || (d->w != d->w) || (d->h != d->h);
}

// 헬퍼 매크로: 텐서 이름으로 데이터 가져오기
#define W(name) weights_get_tensor_data(&weights, name)

// YOLOv5n 파라미터
#define INPUT_SIZE 640
#define NUM_CLASSES 80
#define CONF_THRESHOLD 0.25f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 300

// Strides for each scale (P3, P4, P5)
static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};

int main(int argc, char* argv[]) {
    printf("=== YOLOv5n Inference Pipeline ===\n\n");
    
    // 1. 이미지 로드
    printf("Loading preprocessed image...\n");
    preprocessed_image_t img;
    if (image_load_from_bin("data/input/preprocessed_image.bin", &img) != 0) {
        fprintf(stderr, "Failed to load preprocessed image\n");
        return 1;
    }
    
    printf("Image loaded: %dx%d (original: %dx%d, scale: %.4f)\n",
           img.w, img.h, img.original_w, img.original_h, img.scale);
    
    if (img.w != INPUT_SIZE || img.h != INPUT_SIZE) {
        fprintf(stderr, "Error: Expected input size %dx%d, got %dx%d\n",
                INPUT_SIZE, INPUT_SIZE, img.w, img.h);
        image_free(&img);
        return 1;
    }
    
    // 2. 가중치 로드
    printf("Loading weights...\n");
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        image_free(&img);
        return 1;
    }
    printf("Weights loaded successfully\n\n");
    
    const int n = 1;  // batch size
    const int c_in = 3;
    const int h_in = INPUT_SIZE;
    const int w_in = INPUT_SIZE;
    
    // ===== 중간 버퍼 할당 (동적 할당 - 큰 크기이므로) =====
    // Layer 0: 320x320x16
    float* l0_out = (float*)malloc(1 * 16 * 320 * 320 * sizeof(float));
    // Layer 1: 160x160x32
    float* l1_out = (float*)malloc(1 * 32 * 160 * 160 * sizeof(float));
    // Layer 2: 160x160x32
    float* l2_out = (float*)malloc(1 * 32 * 160 * 160 * sizeof(float));
    // Layer 3: 80x80x64
    float* l3_out = (float*)malloc(1 * 64 * 80 * 80 * sizeof(float));
    // Layer 4: 80x80x64
    float* l4_out = (float*)malloc(1 * 64 * 80 * 80 * sizeof(float));
    // Layer 5: 40x40x128
    float* l5_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 6: 40x40x128
    float* l6_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 7: 20x20x256
    float* l7_out = (float*)malloc(1 * 256 * 20 * 20 * sizeof(float));
    // Layer 8: 20x20x256
    float* l8_out = (float*)malloc(1 * 256 * 20 * 20 * sizeof(float));
    // Layer 9: 20x20x256
    float* l9_out = (float*)malloc(1 * 256 * 20 * 20 * sizeof(float));
    // Layer 10: 20x20x128
    float* l10_out = (float*)malloc(1 * 128 * 20 * 20 * sizeof(float));
    // Layer 11: 40x40x128
    float* l11_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 12: 40x40x256
    float* l12_out = (float*)malloc(1 * 256 * 40 * 40 * sizeof(float));
    // Layer 13: 40x40x128 (P4)
    float* l13_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 14: 40x40x64
    float* l14_out = (float*)malloc(1 * 64 * 40 * 40 * sizeof(float));
    // Layer 15: 80x80x64
    float* l15_out = (float*)malloc(1 * 64 * 80 * 80 * sizeof(float));
    // Layer 16: 80x80x128
    float* l16_out = (float*)malloc(1 * 128 * 80 * 80 * sizeof(float));
    // Layer 17: 80x80x64 (P3)
    float* l17_out = (float*)malloc(1 * 64 * 80 * 80 * sizeof(float));
    // Layer 18: 40x40x64
    float* l18_out = (float*)malloc(1 * 64 * 40 * 40 * sizeof(float));
    // Layer 19: 40x40x128
    float* l19_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 20: 40x40x128
    float* l20_out = (float*)malloc(1 * 128 * 40 * 40 * sizeof(float));
    // Layer 21: 20x20x128
    float* l21_out = (float*)malloc(1 * 128 * 20 * 20 * sizeof(float));
    // Layer 22: 20x20x256
    float* l22_out = (float*)malloc(1 * 256 * 20 * 20 * sizeof(float));
    // Layer 23: 20x20x256 (P5)
    float* l23_out = (float*)malloc(1 * 256 * 20 * 20 * sizeof(float));
    
    // Detect Head 출력 버퍼
    // P3: 80x80 -> cv2: 80x80x64, cv3: 80x80x80
    float* p3_cv2 = (float*)malloc(1 * 64 * 80 * 80 * sizeof(float));
    float* p3_cv3 = (float*)malloc(1 * 80 * 80 * 80 * sizeof(float));
    // P4: 40x40 -> cv2: 40x40x64, cv3: 40x40x80
    float* p4_cv2 = (float*)malloc(1 * 64 * 40 * 40 * sizeof(float));
    float* p4_cv3 = (float*)malloc(1 * 80 * 40 * 40 * sizeof(float));
    // P5: 20x20 -> cv2: 20x20x64, cv3: 20x20x80
    float* p5_cv2 = (float*)malloc(1 * 64 * 20 * 20 * sizeof(float));
    float* p5_cv3 = (float*)malloc(1 * 80 * 20 * 20 * sizeof(float));
    
    // 메모리 할당 확인
    if (!l0_out || !l1_out || !l2_out || !l3_out || !l4_out || !l5_out || 
        !l6_out || !l7_out || !l8_out || !l9_out || !l10_out || !l11_out ||
        !l12_out || !l13_out || !l14_out || !l15_out || !l16_out || !l17_out ||
        !l18_out || !l19_out || !l20_out || !l21_out || !l22_out || !l23_out ||
        !p3_cv2 || !p3_cv3 || !p4_cv2 || !p4_cv3 || !p5_cv2 || !p5_cv3) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }
    
    printf("Running inference pipeline...\n");
    
    // ===== Backbone: Layer 0~9 =====
    printf("Layer 0: Conv(3->16, k=6, s=2, p=2)...\n");
    conv_block_nchw_f32(
        img.data, n, c_in, h_in, w_in,
        W("model.0.conv.weight"), 16, 6, 6,
        2, 2, 2, 2,
        W("model.0.bn.weight"), W("model.0.bn.bias"),
        W("model.0.bn.running_mean"), W("model.0.bn.running_var"),
        1e-3f,
        l0_out, 320, 320);
    
    printf("Layer 1: Conv(16->32, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l0_out, n, 16, 320, 320,
        W("model.1.conv.weight"), 32, 3, 3,
        2, 2, 1, 1,
        W("model.1.bn.weight"), W("model.1.bn.bias"),
        W("model.1.bn.running_mean"), W("model.1.bn.running_var"),
        1e-3f,
        l1_out, 160, 160);
    
    printf("Layer 2: C3(32->32, n=1)...\n");
    const float* l2_m_cv1_w[1] = { W("model.2.m.0.cv1.conv.weight") };
    const float* l2_m_cv1_g[1] = { W("model.2.m.0.cv1.bn.weight") };
    const float* l2_m_cv1_b[1] = { W("model.2.m.0.cv1.bn.bias") };
    const float* l2_m_cv1_m[1] = { W("model.2.m.0.cv1.bn.running_mean") };
    const float* l2_m_cv1_v[1] = { W("model.2.m.0.cv1.bn.running_var") };
    const float* l2_m_cv2_w[1] = { W("model.2.m.0.cv2.conv.weight") };
    const float* l2_m_cv2_g[1] = { W("model.2.m.0.cv2.bn.weight") };
    const float* l2_m_cv2_b[1] = { W("model.2.m.0.cv2.bn.bias") };
    const float* l2_m_cv2_m[1] = { W("model.2.m.0.cv2.bn.running_mean") };
    const float* l2_m_cv2_v[1] = { W("model.2.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l1_out, n, 32, 160, 160,
        W("model.2.cv1.conv.weight"), 16,
        W("model.2.cv1.bn.weight"), W("model.2.cv1.bn.bias"),
        W("model.2.cv1.bn.running_mean"), W("model.2.cv1.bn.running_var"),
        W("model.2.cv2.conv.weight"), 16,
        W("model.2.cv2.bn.weight"), W("model.2.cv2.bn.bias"),
        W("model.2.cv2.bn.running_mean"), W("model.2.cv2.bn.running_var"),
        W("model.2.cv3.conv.weight"), 32,
        W("model.2.cv3.bn.weight"), W("model.2.cv3.bn.bias"),
        W("model.2.cv3.bn.running_mean"), W("model.2.cv3.bn.running_var"),
        1,
        l2_m_cv1_w, l2_m_cv1_g, l2_m_cv1_b, l2_m_cv1_m, l2_m_cv1_v,
        l2_m_cv2_w, l2_m_cv2_g, l2_m_cv2_b, l2_m_cv2_m, l2_m_cv2_v,
        1e-3f,
        l2_out);
    
    printf("Layer 3: Conv(32->64, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l2_out, n, 32, 160, 160,
        W("model.3.conv.weight"), 64, 3, 3,
        2, 2, 1, 1,
        W("model.3.bn.weight"), W("model.3.bn.bias"),
        W("model.3.bn.running_mean"), W("model.3.bn.running_var"),
        1e-3f,
        l3_out, 80, 80);
    
    printf("Layer 4: C3(64->64, n=2)...\n");
    const float* l4_m_cv1_w[2] = { W("model.4.m.0.cv1.conv.weight"), W("model.4.m.1.cv1.conv.weight") };
    const float* l4_m_cv1_g[2] = { W("model.4.m.0.cv1.bn.weight"), W("model.4.m.1.cv1.bn.weight") };
    const float* l4_m_cv1_b[2] = { W("model.4.m.0.cv1.bn.bias"), W("model.4.m.1.cv1.bn.bias") };
    const float* l4_m_cv1_m[2] = { W("model.4.m.0.cv1.bn.running_mean"), W("model.4.m.1.cv1.bn.running_mean") };
    const float* l4_m_cv1_v[2] = { W("model.4.m.0.cv1.bn.running_var"), W("model.4.m.1.cv1.bn.running_var") };
    const float* l4_m_cv2_w[2] = { W("model.4.m.0.cv2.conv.weight"), W("model.4.m.1.cv2.conv.weight") };
    const float* l4_m_cv2_g[2] = { W("model.4.m.0.cv2.bn.weight"), W("model.4.m.1.cv2.bn.weight") };
    const float* l4_m_cv2_b[2] = { W("model.4.m.0.cv2.bn.bias"), W("model.4.m.1.cv2.bn.bias") };
    const float* l4_m_cv2_m[2] = { W("model.4.m.0.cv2.bn.running_mean"), W("model.4.m.1.cv2.bn.running_mean") };
    const float* l4_m_cv2_v[2] = { W("model.4.m.0.cv2.bn.running_var"), W("model.4.m.1.cv2.bn.running_var") };
    c3_nchw_f32(
        l3_out, n, 64, 80, 80,
        W("model.4.cv1.conv.weight"), 32,
        W("model.4.cv1.bn.weight"), W("model.4.cv1.bn.bias"),
        W("model.4.cv1.bn.running_mean"), W("model.4.cv1.bn.running_var"),
        W("model.4.cv2.conv.weight"), 32,
        W("model.4.cv2.bn.weight"), W("model.4.cv2.bn.bias"),
        W("model.4.cv2.bn.running_mean"), W("model.4.cv2.bn.running_var"),
        W("model.4.cv3.conv.weight"), 64,
        W("model.4.cv3.bn.weight"), W("model.4.cv3.bn.bias"),
        W("model.4.cv3.bn.running_mean"), W("model.4.cv3.bn.running_var"),
        2,
        l4_m_cv1_w, l4_m_cv1_g, l4_m_cv1_b, l4_m_cv1_m, l4_m_cv1_v,
        l4_m_cv2_w, l4_m_cv2_g, l4_m_cv2_b, l4_m_cv2_m, l4_m_cv2_v,
        1e-3f,
        l4_out);
    
    printf("Layer 5: Conv(64->128, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l4_out, n, 64, 80, 80,
        W("model.5.conv.weight"), 128, 3, 3,
        2, 2, 1, 1,
        W("model.5.bn.weight"), W("model.5.bn.bias"),
        W("model.5.bn.running_mean"), W("model.5.bn.running_var"),
        1e-3f,
        l5_out, 40, 40);
    
    printf("Layer 6: C3(128->128, n=3)...\n");
    const float* l6_m_cv1_w[3] = { W("model.6.m.0.cv1.conv.weight"), W("model.6.m.1.cv1.conv.weight"), W("model.6.m.2.cv1.conv.weight") };
    const float* l6_m_cv1_g[3] = { W("model.6.m.0.cv1.bn.weight"), W("model.6.m.1.cv1.bn.weight"), W("model.6.m.2.cv1.bn.weight") };
    const float* l6_m_cv1_b[3] = { W("model.6.m.0.cv1.bn.bias"), W("model.6.m.1.cv1.bn.bias"), W("model.6.m.2.cv1.bn.bias") };
    const float* l6_m_cv1_m[3] = { W("model.6.m.0.cv1.bn.running_mean"), W("model.6.m.1.cv1.bn.running_mean"), W("model.6.m.2.cv1.bn.running_mean") };
    const float* l6_m_cv1_v[3] = { W("model.6.m.0.cv1.bn.running_var"), W("model.6.m.1.cv1.bn.running_var"), W("model.6.m.2.cv1.bn.running_var") };
    const float* l6_m_cv2_w[3] = { W("model.6.m.0.cv2.conv.weight"), W("model.6.m.1.cv2.conv.weight"), W("model.6.m.2.cv2.conv.weight") };
    const float* l6_m_cv2_g[3] = { W("model.6.m.0.cv2.bn.weight"), W("model.6.m.1.cv2.bn.weight"), W("model.6.m.2.cv2.bn.weight") };
    const float* l6_m_cv2_b[3] = { W("model.6.m.0.cv2.bn.bias"), W("model.6.m.1.cv2.bn.bias"), W("model.6.m.2.cv2.bn.bias") };
    const float* l6_m_cv2_m[3] = { W("model.6.m.0.cv2.bn.running_mean"), W("model.6.m.1.cv2.bn.running_mean"), W("model.6.m.2.cv2.bn.running_mean") };
    const float* l6_m_cv2_v[3] = { W("model.6.m.0.cv2.bn.running_var"), W("model.6.m.1.cv2.bn.running_var"), W("model.6.m.2.cv2.bn.running_var") };
    c3_nchw_f32(
        l5_out, n, 128, 40, 40,
        W("model.6.cv1.conv.weight"), 64,
        W("model.6.cv1.bn.weight"), W("model.6.cv1.bn.bias"),
        W("model.6.cv1.bn.running_mean"), W("model.6.cv1.bn.running_var"),
        W("model.6.cv2.conv.weight"), 64,
        W("model.6.cv2.bn.weight"), W("model.6.cv2.bn.bias"),
        W("model.6.cv2.bn.running_mean"), W("model.6.cv2.bn.running_var"),
        W("model.6.cv3.conv.weight"), 128,
        W("model.6.cv3.bn.weight"), W("model.6.cv3.bn.bias"),
        W("model.6.cv3.bn.running_mean"), W("model.6.cv3.bn.running_var"),
        3,
        l6_m_cv1_w, l6_m_cv1_g, l6_m_cv1_b, l6_m_cv1_m, l6_m_cv1_v,
        l6_m_cv2_w, l6_m_cv2_g, l6_m_cv2_b, l6_m_cv2_m, l6_m_cv2_v,
        1e-3f,
        l6_out);
    
    printf("Layer 7: Conv(128->256, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l6_out, n, 128, 40, 40,
        W("model.7.conv.weight"), 256, 3, 3,
        2, 2, 1, 1,
        W("model.7.bn.weight"), W("model.7.bn.bias"),
        W("model.7.bn.running_mean"), W("model.7.bn.running_var"),
        1e-3f,
        l7_out, 20, 20);
    
    printf("Layer 8: C3(256->256, n=1)...\n");
    const float* l8_m_cv1_w[1] = { W("model.8.m.0.cv1.conv.weight") };
    const float* l8_m_cv1_g[1] = { W("model.8.m.0.cv1.bn.weight") };
    const float* l8_m_cv1_b[1] = { W("model.8.m.0.cv1.bn.bias") };
    const float* l8_m_cv1_m[1] = { W("model.8.m.0.cv1.bn.running_mean") };
    const float* l8_m_cv1_v[1] = { W("model.8.m.0.cv1.bn.running_var") };
    const float* l8_m_cv2_w[1] = { W("model.8.m.0.cv2.conv.weight") };
    const float* l8_m_cv2_g[1] = { W("model.8.m.0.cv2.bn.weight") };
    const float* l8_m_cv2_b[1] = { W("model.8.m.0.cv2.bn.bias") };
    const float* l8_m_cv2_m[1] = { W("model.8.m.0.cv2.bn.running_mean") };
    const float* l8_m_cv2_v[1] = { W("model.8.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l7_out, n, 256, 20, 20,
        W("model.8.cv1.conv.weight"), 128,
        W("model.8.cv1.bn.weight"), W("model.8.cv1.bn.bias"),
        W("model.8.cv1.bn.running_mean"), W("model.8.cv1.bn.running_var"),
        W("model.8.cv2.conv.weight"), 128,
        W("model.8.cv2.bn.weight"), W("model.8.cv2.bn.bias"),
        W("model.8.cv2.bn.running_mean"), W("model.8.cv2.bn.running_var"),
        W("model.8.cv3.conv.weight"), 256,
        W("model.8.cv3.bn.weight"), W("model.8.cv3.bn.bias"),
        W("model.8.cv3.bn.running_mean"), W("model.8.cv3.bn.running_var"),
        1,
        l8_m_cv1_w, l8_m_cv1_g, l8_m_cv1_b, l8_m_cv1_m, l8_m_cv1_v,
        l8_m_cv2_w, l8_m_cv2_g, l8_m_cv2_b, l8_m_cv2_m, l8_m_cv2_v,
        1e-3f,
        l8_out);
    
    printf("Layer 9: SPPF(256->256)...\n");
    sppf_nchw_f32(
        l8_out, n, 256, 20, 20,
        W("model.9.cv1.conv.weight"), 128,
        W("model.9.cv1.bn.weight"), W("model.9.cv1.bn.bias"),
        W("model.9.cv1.bn.running_mean"), W("model.9.cv1.bn.running_var"),
        W("model.9.cv2.conv.weight"), 256,
        W("model.9.cv2.bn.weight"), W("model.9.cv2.bn.bias"),
        W("model.9.cv2.bn.running_mean"), W("model.9.cv2.bn.running_var"),
        5,
        1e-3f,
        l9_out);
    
    // ===== Neck: Layer 10~23 =====
    printf("Layer 10: Conv(256->128, 1x1)...\n");
    conv_block_nchw_f32(
        l9_out, n, 256, 20, 20,
        W("model.10.conv.weight"), 128, 1, 1,
        1, 1, 0, 0,
        W("model.10.bn.weight"), W("model.10.bn.bias"),
        W("model.10.bn.running_mean"), W("model.10.bn.running_var"),
        1e-3f,
        l10_out, 20, 20);
    
    printf("Layer 11: Upsample...\n");
    upsample_nearest2x_nchw_f32(l10_out, n, 128, 20, 20, l11_out);
    
    printf("Layer 12: Concat(11, 6)...\n");
    concat_nchw_f32(l11_out, 128, l6_out, 128, n, 40, 40, l12_out);
    
    printf("Layer 13: C3(256->128, n=1)...\n");
    const float* l13_m_cv1_w[1] = { W("model.13.m.0.cv1.conv.weight") };
    const float* l13_m_cv1_g[1] = { W("model.13.m.0.cv1.bn.weight") };
    const float* l13_m_cv1_b[1] = { W("model.13.m.0.cv1.bn.bias") };
    const float* l13_m_cv1_m[1] = { W("model.13.m.0.cv1.bn.running_mean") };
    const float* l13_m_cv1_v[1] = { W("model.13.m.0.cv1.bn.running_var") };
    const float* l13_m_cv2_w[1] = { W("model.13.m.0.cv2.conv.weight") };
    const float* l13_m_cv2_g[1] = { W("model.13.m.0.cv2.bn.weight") };
    const float* l13_m_cv2_b[1] = { W("model.13.m.0.cv2.bn.bias") };
    const float* l13_m_cv2_m[1] = { W("model.13.m.0.cv2.bn.running_mean") };
    const float* l13_m_cv2_v[1] = { W("model.13.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l12_out, n, 256, 40, 40,
        W("model.13.cv1.conv.weight"), 128,
        W("model.13.cv1.bn.weight"), W("model.13.cv1.bn.bias"),
        W("model.13.cv1.bn.running_mean"), W("model.13.cv1.bn.running_var"),
        W("model.13.cv2.conv.weight"), 128,
        W("model.13.cv2.bn.weight"), W("model.13.cv2.bn.bias"),
        W("model.13.cv2.bn.running_mean"), W("model.13.cv2.bn.running_var"),
        W("model.13.cv3.conv.weight"), 128,
        W("model.13.cv3.bn.weight"), W("model.13.cv3.bn.bias"),
        W("model.13.cv3.bn.running_mean"), W("model.13.cv3.bn.running_var"),
        1,
        l13_m_cv1_w, l13_m_cv1_g, l13_m_cv1_b, l13_m_cv1_m, l13_m_cv1_v,
        l13_m_cv2_w, l13_m_cv2_g, l13_m_cv2_b, l13_m_cv2_m, l13_m_cv2_v,
        1e-3f,
        l13_out);
    
    printf("Layer 14: Conv(128->64, 1x1)...\n");
    conv_block_nchw_f32(
        l13_out, n, 128, 40, 40,
        W("model.14.conv.weight"), 64, 1, 1,
        1, 1, 0, 0,
        W("model.14.bn.weight"), W("model.14.bn.bias"),
        W("model.14.bn.running_mean"), W("model.14.bn.running_var"),
        1e-3f,
        l14_out, 40, 40);
    
    printf("Layer 15: Upsample...\n");
    upsample_nearest2x_nchw_f32(l14_out, n, 64, 40, 40, l15_out);
    
    printf("Layer 16: Concat(15, 4)...\n");
    concat_nchw_f32(l15_out, 64, l4_out, 64, n, 80, 80, l16_out);
    
    printf("Layer 17: C3(128->64, n=1)...\n");
    const float* l17_m_cv1_w[1] = { W("model.17.m.0.cv1.conv.weight") };
    const float* l17_m_cv1_g[1] = { W("model.17.m.0.cv1.bn.weight") };
    const float* l17_m_cv1_b[1] = { W("model.17.m.0.cv1.bn.bias") };
    const float* l17_m_cv1_m[1] = { W("model.17.m.0.cv1.bn.running_mean") };
    const float* l17_m_cv1_v[1] = { W("model.17.m.0.cv1.bn.running_var") };
    const float* l17_m_cv2_w[1] = { W("model.17.m.0.cv2.conv.weight") };
    const float* l17_m_cv2_g[1] = { W("model.17.m.0.cv2.bn.weight") };
    const float* l17_m_cv2_b[1] = { W("model.17.m.0.cv2.bn.bias") };
    const float* l17_m_cv2_m[1] = { W("model.17.m.0.cv2.bn.running_mean") };
    const float* l17_m_cv2_v[1] = { W("model.17.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l16_out, n, 128, 80, 80,
        W("model.17.cv1.conv.weight"), 64,
        W("model.17.cv1.bn.weight"), W("model.17.cv1.bn.bias"),
        W("model.17.cv1.bn.running_mean"), W("model.17.cv1.bn.running_var"),
        W("model.17.cv2.conv.weight"), 64,
        W("model.17.cv2.bn.weight"), W("model.17.cv2.bn.bias"),
        W("model.17.cv2.bn.running_mean"), W("model.17.cv2.bn.running_var"),
        W("model.17.cv3.conv.weight"), 64,
        W("model.17.cv3.bn.weight"), W("model.17.cv3.bn.bias"),
        W("model.17.cv3.bn.running_mean"), W("model.17.cv3.bn.running_var"),
        1,
        l17_m_cv1_w, l17_m_cv1_g, l17_m_cv1_b, l17_m_cv1_m, l17_m_cv1_v,
        l17_m_cv2_w, l17_m_cv2_g, l17_m_cv2_b, l17_m_cv2_m, l17_m_cv2_v,
        1e-3f,
        l17_out);  // P3
    
    printf("Layer 18: Conv(64->64, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l17_out, n, 64, 80, 80,
        W("model.18.conv.weight"), 64, 3, 3,
        2, 2, 1, 1,
        W("model.18.bn.weight"), W("model.18.bn.bias"),
        W("model.18.bn.running_mean"), W("model.18.bn.running_var"),
        1e-3f,
        l18_out, 40, 40);
    
    printf("Layer 19: Concat(18, 14)...\n");
    concat_nchw_f32(l18_out, 64, l14_out, 64, n, 40, 40, l19_out);
    
    printf("Layer 20: C3(128->128, n=1)...\n");
    const float* l20_m_cv1_w[1] = { W("model.20.m.0.cv1.conv.weight") };
    const float* l20_m_cv1_g[1] = { W("model.20.m.0.cv1.bn.weight") };
    const float* l20_m_cv1_b[1] = { W("model.20.m.0.cv1.bn.bias") };
    const float* l20_m_cv1_m[1] = { W("model.20.m.0.cv1.bn.running_mean") };
    const float* l20_m_cv1_v[1] = { W("model.20.m.0.cv1.bn.running_var") };
    const float* l20_m_cv2_w[1] = { W("model.20.m.0.cv2.conv.weight") };
    const float* l20_m_cv2_g[1] = { W("model.20.m.0.cv2.bn.weight") };
    const float* l20_m_cv2_b[1] = { W("model.20.m.0.cv2.bn.bias") };
    const float* l20_m_cv2_m[1] = { W("model.20.m.0.cv2.bn.running_mean") };
    const float* l20_m_cv2_v[1] = { W("model.20.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l19_out, n, 128, 40, 40,
        W("model.20.cv1.conv.weight"), 64,
        W("model.20.cv1.bn.weight"), W("model.20.cv1.bn.bias"),
        W("model.20.cv1.bn.running_mean"), W("model.20.cv1.bn.running_var"),
        W("model.20.cv2.conv.weight"), 64,
        W("model.20.cv2.bn.weight"), W("model.20.cv2.bn.bias"),
        W("model.20.cv2.bn.running_mean"), W("model.20.cv2.bn.running_var"),
        W("model.20.cv3.conv.weight"), 128,
        W("model.20.cv3.bn.weight"), W("model.20.cv3.bn.bias"),
        W("model.20.cv3.bn.running_mean"), W("model.20.cv3.bn.running_var"),
        1,
        l20_m_cv1_w, l20_m_cv1_g, l20_m_cv1_b, l20_m_cv1_m, l20_m_cv1_v,
        l20_m_cv2_w, l20_m_cv2_g, l20_m_cv2_b, l20_m_cv2_m, l20_m_cv2_v,
        1e-3f,
        l20_out);  // P4
    
    printf("Layer 21: Conv(128->128, k=3, s=2, p=1)...\n");
    conv_block_nchw_f32(
        l20_out, n, 128, 40, 40,
        W("model.21.conv.weight"), 128, 3, 3,
        2, 2, 1, 1,
        W("model.21.bn.weight"), W("model.21.bn.bias"),
        W("model.21.bn.running_mean"), W("model.21.bn.running_var"),
        1e-3f,
        l21_out, 20, 20);
    
    printf("Layer 22: Concat(21, 10)...\n");
    concat_nchw_f32(l21_out, 128, l10_out, 128, n, 20, 20, l22_out);
    
    printf("Layer 23: C3(256->256, n=1)...\n");
    const float* l23_m_cv1_w[1] = { W("model.23.m.0.cv1.conv.weight") };
    const float* l23_m_cv1_g[1] = { W("model.23.m.0.cv1.bn.weight") };
    const float* l23_m_cv1_b[1] = { W("model.23.m.0.cv1.bn.bias") };
    const float* l23_m_cv1_m[1] = { W("model.23.m.0.cv1.bn.running_mean") };
    const float* l23_m_cv1_v[1] = { W("model.23.m.0.cv1.bn.running_var") };
    const float* l23_m_cv2_w[1] = { W("model.23.m.0.cv2.conv.weight") };
    const float* l23_m_cv2_g[1] = { W("model.23.m.0.cv2.bn.weight") };
    const float* l23_m_cv2_b[1] = { W("model.23.m.0.cv2.bn.bias") };
    const float* l23_m_cv2_m[1] = { W("model.23.m.0.cv2.bn.running_mean") };
    const float* l23_m_cv2_v[1] = { W("model.23.m.0.cv2.bn.running_var") };
    c3_nchw_f32(
        l22_out, n, 256, 20, 20,
        W("model.23.cv1.conv.weight"), 128,
        W("model.23.cv1.bn.weight"), W("model.23.cv1.bn.bias"),
        W("model.23.cv1.bn.running_mean"), W("model.23.cv1.bn.running_var"),
        W("model.23.cv2.conv.weight"), 128,
        W("model.23.cv2.bn.weight"), W("model.23.cv2.bn.bias"),
        W("model.23.cv2.bn.running_mean"), W("model.23.cv2.bn.running_var"),
        W("model.23.cv3.conv.weight"), 256,
        W("model.23.cv3.bn.weight"), W("model.23.cv3.bn.bias"),
        W("model.23.cv3.bn.running_mean"), W("model.23.cv3.bn.running_var"),
        1,
        l23_m_cv1_w, l23_m_cv1_g, l23_m_cv1_b, l23_m_cv1_m, l23_m_cv1_v,
        l23_m_cv2_w, l23_m_cv2_g, l23_m_cv2_b, l23_m_cv2_m, l23_m_cv2_v,
        1e-3f,
        l23_out);  // P5
    
    // ===== Head: Layer 24 (Detect) =====
    printf("Layer 24: Detect Head...\n");
    detect_head_nchw_f32(
        // P3 입력 및 cv2, cv3 파라미터
        l17_out, 64, 80, 80,
        W("model.24.cv2.0.0.conv.weight"), 64,
        W("model.24.cv2.0.0.bn.weight"), W("model.24.cv2.0.0.bn.bias"),
        W("model.24.cv2.0.0.bn.running_mean"), W("model.24.cv2.0.0.bn.running_var"),
        W("model.24.cv2.0.1.conv.weight"),
        W("model.24.cv2.0.1.bn.weight"), W("model.24.cv2.0.1.bn.bias"),
        W("model.24.cv2.0.1.bn.running_mean"), W("model.24.cv2.0.1.bn.running_var"),
        W("model.24.cv2.0.2.weight"), W("model.24.cv2.0.2.bias"),
        W("model.24.cv3.0.0.conv.weight"), 80,
        W("model.24.cv3.0.0.bn.weight"), W("model.24.cv3.0.0.bn.bias"),
        W("model.24.cv3.0.0.bn.running_mean"), W("model.24.cv3.0.0.bn.running_var"),
        W("model.24.cv3.0.1.conv.weight"),
        W("model.24.cv3.0.1.bn.weight"), W("model.24.cv3.0.1.bn.bias"),
        W("model.24.cv3.0.1.bn.running_mean"), W("model.24.cv3.0.1.bn.running_var"),
        W("model.24.cv3.0.2.weight"), W("model.24.cv3.0.2.bias"),
        // P4 입력 및 cv2, cv3 파라미터
        l20_out, 128, 40, 40,
        W("model.24.cv2.1.0.conv.weight"), 64,
        W("model.24.cv2.1.0.bn.weight"), W("model.24.cv2.1.0.bn.bias"),
        W("model.24.cv2.1.0.bn.running_mean"), W("model.24.cv2.1.0.bn.running_var"),
        W("model.24.cv2.1.1.conv.weight"),
        W("model.24.cv2.1.1.bn.weight"), W("model.24.cv2.1.1.bn.bias"),
        W("model.24.cv2.1.1.bn.running_mean"), W("model.24.cv2.1.1.bn.running_var"),
        W("model.24.cv2.1.2.weight"), W("model.24.cv2.1.2.bias"),
        W("model.24.cv3.1.0.conv.weight"), 80,
        W("model.24.cv3.1.0.bn.weight"), W("model.24.cv3.1.0.bn.bias"),
        W("model.24.cv3.1.0.bn.running_mean"), W("model.24.cv3.1.0.bn.running_var"),
        W("model.24.cv3.1.1.conv.weight"),
        W("model.24.cv3.1.1.bn.weight"), W("model.24.cv3.1.1.bn.bias"),
        W("model.24.cv3.1.1.bn.running_mean"), W("model.24.cv3.1.1.bn.running_var"),
        W("model.24.cv3.1.2.weight"), W("model.24.cv3.1.2.bias"),
        // P5 입력 및 cv2, cv3 파라미터
        l23_out, 256, 20, 20,
        W("model.24.cv2.2.0.conv.weight"), 64,
        W("model.24.cv2.2.0.bn.weight"), W("model.24.cv2.2.0.bn.bias"),
        W("model.24.cv2.2.0.bn.running_mean"), W("model.24.cv2.2.0.bn.running_var"),
        W("model.24.cv2.2.1.conv.weight"),
        W("model.24.cv2.2.1.bn.weight"), W("model.24.cv2.2.1.bn.bias"),
        W("model.24.cv2.2.1.bn.running_mean"), W("model.24.cv2.2.1.bn.running_var"),
        W("model.24.cv2.2.2.weight"), W("model.24.cv2.2.2.bias"),
        W("model.24.cv3.2.0.conv.weight"), 80,
        W("model.24.cv3.2.0.bn.weight"), W("model.24.cv3.2.0.bn.bias"),
        W("model.24.cv3.2.0.bn.running_mean"), W("model.24.cv3.2.0.bn.running_var"),
        W("model.24.cv3.2.1.conv.weight"),
        W("model.24.cv3.2.1.bn.weight"), W("model.24.cv3.2.1.bn.bias"),
        W("model.24.cv3.2.1.bn.running_mean"), W("model.24.cv3.2.1.bn.running_var"),
        W("model.24.cv3.2.2.weight"), W("model.24.cv3.2.2.bias"),
        1e-3f,
        // 출력 (cv2, cv3 각각)
        p3_cv2, 80, 80,
        p3_cv3, 80, 80,
        p4_cv2, 40, 40,
        p4_cv3, 40, 40,
        p5_cv2, 20, 20,
        p5_cv3, 20, 20);
    
    // ===== Decode =====
    printf("Decoding detections...\n");
    detection_t* decoded_detections = (detection_t*)malloc(MAX_DETECTIONS * sizeof(detection_t));
    if (!decoded_detections) {
        fprintf(stderr, "Memory allocation failed for decoded detections\n");
        goto cleanup;
    }
    
    int32_t num_decoded = decode_detections_nchw_f32(
        // P3 cv2, cv3 출력
        p3_cv2, 80, 80,
        p3_cv3, 80,
        // P4 cv2, cv3 출력
        p4_cv2, 40, 40,
        p4_cv3, 80,
        // P5 cv2, cv3 출력
        p5_cv2, 20, 20,
        p5_cv3, 80,
        // 파라미터
        NUM_CLASSES,
        CONF_THRESHOLD,
        INPUT_SIZE,
        STRIDES,
        // 출력
        decoded_detections,
        MAX_DETECTIONS);
    
    printf("Decoded %d detections (before NMS)\n", num_decoded);
    
    /* NaN bbox 제거 (Decode/DFL 연산 잔여 수치 이슈) */
    {
        int32_t w = 0;
        for (int32_t i = 0; i < num_decoded; i++) {
            if (!detection_has_nan_bbox(&decoded_detections[i])) {
                if (w != i) decoded_detections[w] = decoded_detections[i];
                w++;
            }
        }
        if (w < num_decoded) {
            printf("Filtered %d detections with NaN bbox -> %d valid\n", num_decoded - w, w);
            num_decoded = w;
        }
    }
    
    // Decode가 0개를 반환한 경우 처리
    if (num_decoded == 0) {
        printf("Warning: No detections found (confidence threshold may be too high)\n");
        printf("Results saved to data/output/detections.txt (empty)\n");
        
        // 빈 결과 파일 생성
        system("mkdir -p data/output");
        FILE* fout = fopen("data/output/detections.txt", "w");
        if (fout) {
            fprintf(fout, "# YOLOv5n Detection Results\n");
            fprintf(fout, "# Input image: %dx%d (original: %dx%d, scale: %.4f)\n",
                    img.w, img.h, img.original_w, img.original_h, img.scale);
            fprintf(fout, "# Total detections: 0\n");
            fprintf(fout, "# No detections found (confidence threshold: %.2f)\n", CONF_THRESHOLD);
            fprintf(fout, "# Format: class_id confidence x y w h (normalized coordinates)\n");
            fclose(fout);
        }
        
        free(decoded_detections);
        goto cleanup;
    }
    
    // ===== NMS =====
    printf("Applying NMS...\n");
    detection_t* nms_detections = NULL;
    int32_t num_nms = 0;
    
    // NMS 전에 confidence로 정렬
    for (int i = 0; i < num_decoded - 1; i++) {
        for (int j = i + 1; j < num_decoded; j++) {
            if (decoded_detections[i].conf < decoded_detections[j].conf) {
                detection_t tmp = decoded_detections[i];
                decoded_detections[i] = decoded_detections[j];
                decoded_detections[j] = tmp;
            }
        }
    }
    
    int nms_ret = nms(
        decoded_detections, num_decoded,
        &nms_detections, &num_nms,
        IOU_THRESHOLD,
        MAX_DETECTIONS);
    
    if (nms_ret != 0) {
        fprintf(stderr, "NMS failed\n");
        free(decoded_detections);
        goto cleanup;
    }
    
    printf("NMS output: %d detections\n\n", num_nms);
    
    // ===== 결과 저장 =====
    printf("Saving results to data/output/detections.txt...\n");
    
    // 출력 디렉토리 생성 (없으면)
    system("mkdir -p data/output");
    
    FILE* fout = fopen("data/output/detections.txt", "w");
    if (!fout) {
        fprintf(stderr, "Failed to open output file\n");
        free(decoded_detections);
        if (nms_detections) free(nms_detections);
        goto cleanup;
    }
    
    // 헤더 정보
    fprintf(fout, "# YOLOv5n Detection Results\n");
    fprintf(fout, "# Input image: %dx%d (original: %dx%d, scale: %.4f)\n",
            img.w, img.h, img.original_w, img.original_h, img.scale);
    fprintf(fout, "# Total detections: %d\n", num_nms);
    fprintf(fout, "# Format: class_id confidence x y w h (normalized coordinates)\n");
    fprintf(fout, "#\n");
    
    // Detection 결과 저장
    for (int i = 0; i < num_nms; i++) {
        fprintf(fout, "%d %.6f %.6f %.6f %.6f %.6f\n",
                nms_detections[i].cls_id,
                nms_detections[i].conf,
                nms_detections[i].x,
                nms_detections[i].y,
                nms_detections[i].w,
                nms_detections[i].h);
    }
    
    fclose(fout);
    
    // 콘솔에도 출력
    printf("\n=== Detection Results ===\n");
    for (int i = 0; i < num_nms && i < 10; i++) {  // 최대 10개만 출력
        printf("[%d] cls=%d conf=%.4f bbox=(%.4f,%.4f,%.4f,%.4f)\n",
               i, nms_detections[i].cls_id, nms_detections[i].conf,
               nms_detections[i].x, nms_detections[i].y,
               nms_detections[i].w, nms_detections[i].h);
    }
    if (num_nms > 10) {
        printf("... (%d more detections)\n", num_nms - 10);
    }
    
    printf("\nResults saved to data/output/detections.txt\n");
    printf("Inference completed successfully!\n");
    
    // 정리
    free(decoded_detections);
    if (nms_detections) free(nms_detections);
    
cleanup:
    // 메모리 해제
    free(l0_out); free(l1_out); free(l2_out); free(l3_out); free(l4_out);
    free(l5_out); free(l6_out); free(l7_out); free(l8_out); free(l9_out);
    free(l10_out); free(l11_out); free(l12_out); free(l13_out); free(l14_out);
    free(l15_out); free(l16_out); free(l17_out); free(l18_out); free(l19_out);
    free(l20_out); free(l21_out); free(l22_out); free(l23_out);
    free(p3_cv2); free(p3_cv3); free(p4_cv2); free(p4_cv3); free(p5_cv2); free(p5_cv3);
    
    weights_free(&weights);
    image_free(&img);
    
    return 0;
}
