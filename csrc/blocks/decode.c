/**
 * Classic YOLOv5n Anchor-based Decode
 * 255ch = 3 * 85. Layout: [anchor0_85, anchor1_85, anchor2_85].
 * base = (a*85)*(H*W) + (y*W+x). x,y,w,h,obj,cls0..79.
 */

#include "decode.h"
#include <math.h>
#include <stdlib.h>

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int32_t decode_nchw_f32(
    const float* p3, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_h, int32_t p5_w,
    int32_t num_classes,
    float conf_threshold,
    int32_t input_size,
    const float strides[3],
    const float anchors[3][6],
    detection_t* detections,
    int32_t max_detections)
{
    int32_t count = 0;
    const int32_t no = 5 + num_classes;  /* 85 */

    for (int scale = 0; scale < 3; scale++) {
        const float* feat = NULL;
        int32_t gh = 0, gw = 0;
        float stride = 0.0f;
        const float* anc = NULL;

        switch (scale) {
            case 0: feat = p3; gh = p3_h; gw = p3_w; stride = strides[0]; anc = anchors[0]; break;
            case 1: feat = p4; gh = p4_h; gw = p4_w; stride = strides[1]; anc = anchors[1]; break;
            case 2: feat = p5; gh = p5_h; gw = p5_w; stride = strides[2]; anc = anchors[2]; break;
        }
        if (!feat) continue;

        const int32_t gsize = gh * gw;

        for (int32_t y = 0; y < gh; y++) {
            for (int32_t x = 0; x < gw; x++) {
                const int32_t spatial = y * gw + x;

                for (int a = 0; a < 3; a++) {
                    const int32_t base = (a * no) * gsize + spatial;

                    float bx = feat[base + 0 * gsize];
                    float by = feat[base + 1 * gsize];
                    float bw = feat[base + 2 * gsize];
                    float bh = feat[base + 3 * gsize];
                    float obj_logit = feat[base + 4 * gsize];

                    float obj_conf = sigmoid_f(obj_logit);
                    float max_cls = 0.0f;
                    int32_t max_cls_id = 0;
                    for (int c = 0; c < num_classes; c++) {
                        float v = sigmoid_f(feat[base + (5 + c) * gsize]);
                        if (v > max_cls) { max_cls = v; max_cls_id = c; }
                    }
                    float conf = obj_conf * max_cls;
                    if (conf < conf_threshold) continue;
                    if (count >= max_detections) goto done;

                    float tx = sigmoid_f(bx);
                    float ty = sigmoid_f(by);
                    float tw = sigmoid_f(bw);
                    float th = sigmoid_f(bh);

                    float gx = (float)x - 0.5f;
                    float gy = (float)y - 0.5f;
                    float cx = (tx * 2.0f + gx) * stride;
                    float cy = (ty * 2.0f + gy) * stride;

                    float aw = anc[a * 2 + 0];
                    float ah = anc[a * 2 + 1];
                    float ww = (tw * 2.0f) * (tw * 2.0f) * aw;
                    float hh = (th * 2.0f) * (th * 2.0f) * ah;

                    detections[count].x = cx / (float)input_size;
                    detections[count].y = cy / (float)input_size;
                    detections[count].w = ww / (float)input_size;
                    detections[count].h = hh / (float)input_size;
                    detections[count].conf = conf;
                    detections[count].cls_id = max_cls_id;
                    count++;
                }
            }
        }
    }
done:
    return count;
}
