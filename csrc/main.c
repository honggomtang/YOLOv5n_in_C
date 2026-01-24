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

#define W(name) weights_get_tensor_data(&weights, name)

#define INPUT_SIZE 640
#define NUM_CLASSES 80
#define CONF_THRESHOLD 0.25f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 300

static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};
static const float ANCHORS[3][6] = {
    {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
    {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
    {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
};

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== YOLOv5n Inference (Fused) ===\n\n");
    
    // 이미지 로드
    preprocessed_image_t img;
    if (image_load_from_bin("data/input/preprocessed_image.bin", &img) != 0) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    printf("Image: %dx%d\n", img.w, img.h);
    
    // 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        image_free(&img);
        return 1;
    }
    printf("Weights: %d tensors\n\n", weights.num_tensors);
    
    const int n = 1;
    
    // 메모리 할당
    float* l0  = malloc(1 * 16  * 320 * 320 * sizeof(float));
    float* l1  = malloc(1 * 32  * 160 * 160 * sizeof(float));
    float* l2  = malloc(1 * 32  * 160 * 160 * sizeof(float));
    float* l3  = malloc(1 * 64  * 80  * 80  * sizeof(float));
    float* l4  = malloc(1 * 64  * 80  * 80  * sizeof(float));
    float* l5  = malloc(1 * 128 * 40  * 40  * sizeof(float));
    float* l6  = malloc(1 * 128 * 40  * 40  * sizeof(float));
    float* l7  = malloc(1 * 256 * 20  * 20  * sizeof(float));
    float* l8  = malloc(1 * 256 * 20  * 20  * sizeof(float));
    float* l9  = malloc(1 * 256 * 20  * 20  * sizeof(float));
    float* l10 = malloc(1 * 128 * 20  * 20  * sizeof(float));
    float* l11 = malloc(1 * 128 * 40  * 40  * sizeof(float));
    float* l12 = malloc(1 * 256 * 40  * 40  * sizeof(float));
    float* l13 = malloc(1 * 128 * 40  * 40  * sizeof(float));
    float* l14 = malloc(1 * 64  * 40  * 40  * sizeof(float));
    float* l15 = malloc(1 * 64  * 80  * 80  * sizeof(float));
    float* l16 = malloc(1 * 128 * 80  * 80  * sizeof(float));
    float* l17 = malloc(1 * 64  * 80  * 80  * sizeof(float));  // P3
    float* l18 = malloc(1 * 64  * 40  * 40  * sizeof(float));
    float* l19 = malloc(1 * 128 * 40  * 40  * sizeof(float));
    float* l20 = malloc(1 * 128 * 40  * 40  * sizeof(float));  // P4
    float* l21 = malloc(1 * 128 * 20  * 20  * sizeof(float));
    float* l22 = malloc(1 * 256 * 20  * 20  * sizeof(float));
    float* l23 = malloc(1 * 256 * 20  * 20  * sizeof(float));  // P5
    float* p3  = malloc(1 * 255 * 80  * 80  * sizeof(float));
    float* p4  = malloc(1 * 255 * 40  * 40  * sizeof(float));
    float* p5  = malloc(1 * 255 * 20  * 20  * sizeof(float));

    printf("Running inference...\n");
    
    // ===== Backbone =====
    // Layer 0: Conv 6x6 s2
    conv_block_nchw_f32(img.data, n, 3, 640, 640,
        W("model.0.conv.weight"), 16, 6, 6, 2, 2, 2, 2,
        W("model.0.conv.bias"), l0, 320, 320);
    

    // Layer 1: Conv 3x3 s2
    conv_block_nchw_f32(l0, n, 16, 320, 320,
        W("model.1.conv.weight"), 32, 3, 3, 2, 2, 1, 1,
        W("model.1.conv.bias"), l1, 160, 160);

    // Layer 2: C3 (n=1)
    const float* l2_cv1w[] = {W("model.2.m.0.cv1.conv.weight")};
    const float* l2_cv1b[] = {W("model.2.m.0.cv1.conv.bias")};
    const float* l2_cv2w[] = {W("model.2.m.0.cv2.conv.weight")};
    const float* l2_cv2b[] = {W("model.2.m.0.cv2.conv.bias")};
    c3_nchw_f32(l1, n, 32, 160, 160,
        W("model.2.cv1.conv.weight"), 16, W("model.2.cv1.conv.bias"),
        W("model.2.cv2.conv.weight"), 16, W("model.2.cv2.conv.bias"),
        W("model.2.cv3.conv.weight"), 32, W("model.2.cv3.conv.bias"),
        1, l2_cv1w, l2_cv1b, l2_cv2w, l2_cv2b, 1, l2);  // shortcut=1

    // Layer 3: Conv 3x3 s2
    conv_block_nchw_f32(l2, n, 32, 160, 160,
        W("model.3.conv.weight"), 64, 3, 3, 2, 2, 1, 1,
        W("model.3.conv.bias"), l3, 80, 80);

    // Layer 4: C3 (n=2)
    const float* l4_cv1w[] = {W("model.4.m.0.cv1.conv.weight"), W("model.4.m.1.cv1.conv.weight")};
    const float* l4_cv1b[] = {W("model.4.m.0.cv1.conv.bias"), W("model.4.m.1.cv1.conv.bias")};
    const float* l4_cv2w[] = {W("model.4.m.0.cv2.conv.weight"), W("model.4.m.1.cv2.conv.weight")};
    const float* l4_cv2b[] = {W("model.4.m.0.cv2.conv.bias"), W("model.4.m.1.cv2.conv.bias")};
    c3_nchw_f32(l3, n, 64, 80, 80,
        W("model.4.cv1.conv.weight"), 32, W("model.4.cv1.conv.bias"),
        W("model.4.cv2.conv.weight"), 32, W("model.4.cv2.conv.bias"),
        W("model.4.cv3.conv.weight"), 64, W("model.4.cv3.conv.bias"),
        2, l4_cv1w, l4_cv1b, l4_cv2w, l4_cv2b, 1, l4);  // shortcut=1

    // Layer 5: Conv 3x3 s2
    conv_block_nchw_f32(l4, n, 64, 80, 80,
        W("model.5.conv.weight"), 128, 3, 3, 2, 2, 1, 1,
        W("model.5.conv.bias"), l5, 40, 40);

    // Layer 6: C3 (n=3)
    const float* l6_cv1w[] = {W("model.6.m.0.cv1.conv.weight"), W("model.6.m.1.cv1.conv.weight"), W("model.6.m.2.cv1.conv.weight")};
    const float* l6_cv1b[] = {W("model.6.m.0.cv1.conv.bias"), W("model.6.m.1.cv1.conv.bias"), W("model.6.m.2.cv1.conv.bias")};
    const float* l6_cv2w[] = {W("model.6.m.0.cv2.conv.weight"), W("model.6.m.1.cv2.conv.weight"), W("model.6.m.2.cv2.conv.weight")};
    const float* l6_cv2b[] = {W("model.6.m.0.cv2.conv.bias"), W("model.6.m.1.cv2.conv.bias"), W("model.6.m.2.cv2.conv.bias")};
    c3_nchw_f32(l5, n, 128, 40, 40,
        W("model.6.cv1.conv.weight"), 64, W("model.6.cv1.conv.bias"),
        W("model.6.cv2.conv.weight"), 64, W("model.6.cv2.conv.bias"),
        W("model.6.cv3.conv.weight"), 128, W("model.6.cv3.conv.bias"),
        3, l6_cv1w, l6_cv1b, l6_cv2w, l6_cv2b, 1, l6);  // shortcut=1

    // Layer 7: Conv 3x3 s2
    conv_block_nchw_f32(l6, n, 128, 40, 40,
        W("model.7.conv.weight"), 256, 3, 3, 2, 2, 1, 1,
        W("model.7.conv.bias"), l7, 20, 20);

    // Layer 8: C3 (n=1)
    const float* l8_cv1w[] = {W("model.8.m.0.cv1.conv.weight")};
    const float* l8_cv1b[] = {W("model.8.m.0.cv1.conv.bias")};
    const float* l8_cv2w[] = {W("model.8.m.0.cv2.conv.weight")};
    const float* l8_cv2b[] = {W("model.8.m.0.cv2.conv.bias")};
    c3_nchw_f32(l7, n, 256, 20, 20,
        W("model.8.cv1.conv.weight"), 128, W("model.8.cv1.conv.bias"),
        W("model.8.cv2.conv.weight"), 128, W("model.8.cv2.conv.bias"),
        W("model.8.cv3.conv.weight"), 256, W("model.8.cv3.conv.bias"),
        1, l8_cv1w, l8_cv1b, l8_cv2w, l8_cv2b, 1, l8);  // shortcut=1

    // Layer 9: SPPF
    sppf_nchw_f32(l8, n, 256, 20, 20,
        W("model.9.cv1.conv.weight"), 128, W("model.9.cv1.conv.bias"),
        W("model.9.cv2.conv.weight"), 256, W("model.9.cv2.conv.bias"),
        5, l9);

    // ===== Neck =====
    // Layer 10: Conv 1x1
    conv_block_nchw_f32(l9, n, 256, 20, 20,
        W("model.10.conv.weight"), 128, 1, 1, 1, 1, 0, 0,
        W("model.10.conv.bias"), l10, 20, 20);

    // Layer 11: Upsample
    upsample_nearest2x_nchw_f32(l10, n, 128, 20, 20, l11);

    // Layer 12: Concat (l11 + l6)
    concat_nchw_f32(l11, 128, l6, 128, n, 40, 40, l12);

    // Layer 13: C3 (n=1)
    const float* l13_cv1w[] = {W("model.13.m.0.cv1.conv.weight")};
    const float* l13_cv1b[] = {W("model.13.m.0.cv1.conv.bias")};
    const float* l13_cv2w[] = {W("model.13.m.0.cv2.conv.weight")};
    const float* l13_cv2b[] = {W("model.13.m.0.cv2.conv.bias")};
    c3_nchw_f32(l12, n, 256, 40, 40,
        W("model.13.cv1.conv.weight"), 64, W("model.13.cv1.conv.bias"),
        W("model.13.cv2.conv.weight"), 64, W("model.13.cv2.conv.bias"),
        W("model.13.cv3.conv.weight"), 128, W("model.13.cv3.conv.bias"),
        1, l13_cv1w, l13_cv1b, l13_cv2w, l13_cv2b, 0, l13);  // shortcut=0 (head)

    // Layer 14: Conv 1x1
    conv_block_nchw_f32(l13, n, 128, 40, 40,
        W("model.14.conv.weight"), 64, 1, 1, 1, 1, 0, 0,
        W("model.14.conv.bias"), l14, 40, 40);

    // Layer 15: Upsample
    upsample_nearest2x_nchw_f32(l14, n, 64, 40, 40, l15);

    // Layer 16: Concat (l15 + l4)
    concat_nchw_f32(l15, 64, l4, 64, n, 80, 80, l16);

    // Layer 17: C3 (n=1) -> P3
    const float* l17_cv1w[] = {W("model.17.m.0.cv1.conv.weight")};
    const float* l17_cv1b[] = {W("model.17.m.0.cv1.conv.bias")};
    const float* l17_cv2w[] = {W("model.17.m.0.cv2.conv.weight")};
    const float* l17_cv2b[] = {W("model.17.m.0.cv2.conv.bias")};
    c3_nchw_f32(l16, n, 128, 80, 80,
        W("model.17.cv1.conv.weight"), 32, W("model.17.cv1.conv.bias"),
        W("model.17.cv2.conv.weight"), 32, W("model.17.cv2.conv.bias"),
        W("model.17.cv3.conv.weight"), 64, W("model.17.cv3.conv.bias"),
        1, l17_cv1w, l17_cv1b, l17_cv2w, l17_cv2b, 0, l17);  // shortcut=0 (head)

    // Layer 18: Conv 3x3 s2
    conv_block_nchw_f32(l17, n, 64, 80, 80,
        W("model.18.conv.weight"), 64, 3, 3, 2, 2, 1, 1,
        W("model.18.conv.bias"), l18, 40, 40);

    // Layer 19: Concat (l18 + l14)
    concat_nchw_f32(l18, 64, l14, 64, n, 40, 40, l19);

    // Layer 20: C3 (n=1) -> P4
    const float* l20_cv1w[] = {W("model.20.m.0.cv1.conv.weight")};
    const float* l20_cv1b[] = {W("model.20.m.0.cv1.conv.bias")};
    const float* l20_cv2w[] = {W("model.20.m.0.cv2.conv.weight")};
    const float* l20_cv2b[] = {W("model.20.m.0.cv2.conv.bias")};
    c3_nchw_f32(l19, n, 128, 40, 40,
        W("model.20.cv1.conv.weight"), 64, W("model.20.cv1.conv.bias"),
        W("model.20.cv2.conv.weight"), 64, W("model.20.cv2.conv.bias"),
        W("model.20.cv3.conv.weight"), 128, W("model.20.cv3.conv.bias"),
        1, l20_cv1w, l20_cv1b, l20_cv2w, l20_cv2b, 0, l20);  // shortcut=0 (head)

    // Layer 21: Conv 3x3 s2
    conv_block_nchw_f32(l20, n, 128, 40, 40,
        W("model.21.conv.weight"), 128, 3, 3, 2, 2, 1, 1,
        W("model.21.conv.bias"), l21, 20, 20);

    // Layer 22: Concat (l21 + l10)
    concat_nchw_f32(l21, 128, l10, 128, n, 20, 20, l22);

    // Layer 23: C3 (n=1) -> P5
    const float* l23_cv1w[] = {W("model.23.m.0.cv1.conv.weight")};
    const float* l23_cv1b[] = {W("model.23.m.0.cv1.conv.bias")};
    const float* l23_cv2w[] = {W("model.23.m.0.cv2.conv.weight")};
    const float* l23_cv2b[] = {W("model.23.m.0.cv2.conv.bias")};
    c3_nchw_f32(l22, n, 256, 20, 20,
        W("model.23.cv1.conv.weight"), 128, W("model.23.cv1.conv.bias"),
        W("model.23.cv2.conv.weight"), 128, W("model.23.cv2.conv.bias"),
        W("model.23.cv3.conv.weight"), 256, W("model.23.cv3.conv.bias"),
        1, l23_cv1w, l23_cv1b, l23_cv2w, l23_cv2b, 0, l23);  // shortcut=0 (head)

    // ===== Detect Head =====
    detect_nchw_f32(
        l17, 64, 80, 80,
        l20, 128, 40, 40,
        l23, 256, 20, 20,
        W("model.24.m.0.weight"), W("model.24.m.0.bias"),
        W("model.24.m.1.weight"), W("model.24.m.1.bias"),
        W("model.24.m.2.weight"), W("model.24.m.2.bias"),
        p3, p4, p5);

    // ===== Decode =====
    detection_t* dets = malloc(MAX_DETECTIONS * sizeof(detection_t));
    int32_t num_dets = decode_nchw_f32(
        p3, 80, 80, p4, 40, 40, p5, 20, 20,
        NUM_CLASSES, CONF_THRESHOLD, INPUT_SIZE, STRIDES, ANCHORS,
        dets, MAX_DETECTIONS);

    printf("Decoded: %d detections\n", num_dets);

    // Sort by confidence
    for (int i = 0; i < num_dets - 1; i++) {
        for (int j = i + 1; j < num_dets; j++) {
            if (dets[i].conf < dets[j].conf) {
                detection_t t = dets[i]; dets[i] = dets[j]; dets[j] = t;
            }
        }
    }

    // NMS
    detection_t* nms_dets = NULL;
    int32_t num_nms = 0;
    nms(dets, num_dets, &nms_dets, &num_nms, IOU_THRESHOLD, MAX_DETECTIONS);

    printf("After NMS: %d detections\n", num_nms);

    // 결과 저장
    FILE* f = fopen("data/output/detections.txt", "w");
    if (f) {
        fprintf(f, "# YOLOv5n Detection Results\n");
        fprintf(f, "# Detections: %d\n", num_nms);
        fprintf(f, "# Format: class_id confidence x y w h\n\n");
        for (int i = 0; i < num_nms; i++) {
            fprintf(f, "%d %.6f %.6f %.6f %.6f %.6f\n",
                nms_dets[i].cls_id, nms_dets[i].conf,
                nms_dets[i].x, nms_dets[i].y, nms_dets[i].w, nms_dets[i].h);
        }
        fclose(f);
        printf("Saved to data/output/detections.txt\n");
    }

    // Cleanup
    free(l0); free(l1); free(l2); free(l3); free(l4); free(l5);
    free(l6); free(l7); free(l8); free(l9); free(l10); free(l11);
    free(l12); free(l13); free(l14); free(l15); free(l16); free(l17);
    free(l18); free(l19); free(l20); free(l21); free(l22); free(l23);
    free(p3); free(p4); free(p5);
    free(dets);
    if (nms_dets) free(nms_dets);
    weights_free(&weights);
    image_free(&img);
    
    return 0;
}
