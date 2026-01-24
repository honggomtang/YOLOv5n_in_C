#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline void safe_read(void* dest, const uint8_t** src, size_t size) {
    memcpy(dest, *src, size);
    *src += size;
}

static int parse_image_data(const uint8_t* ptr, size_t data_len, preprocessed_image_t* img) {
    const uint8_t* curr = ptr;
    const uint8_t* end = ptr + data_len;

    // 헤더 크기: 6 * 4 bytes = 24 bytes
    if (curr + 24 > end) return -1;

    uint32_t original_w, original_h, size;
    float scale;
    uint32_t pad_x, pad_y;
    
    safe_read(&original_w, &curr, 4);
    safe_read(&original_h, &curr, 4);
    safe_read(&scale, &curr, 4);
    safe_read(&pad_x, &curr, 4);
    safe_read(&pad_y, &curr, 4);
    safe_read(&size, &curr, 4);
    
    img->original_w = (int32_t)original_w;
    img->original_h = (int32_t)original_h;
    img->scale = scale;
    img->pad_x = (int32_t)pad_x;
    img->pad_y = (int32_t)pad_y;
    img->c = 3;
    img->h = (int32_t)size;
    img->w = (int32_t)size;
    
    // 데이터 복사
    size_t data_bytes = 3 * size * size * sizeof(float);
    if (curr + data_bytes > end) return -1;

    img->data = (float*)malloc(data_bytes);
    if (!img->data) return -1;

    safe_read(img->data, &curr, data_bytes);
    return 0;
}

int image_init_from_memory(uintptr_t base_addr, preprocessed_image_t* img) {
    return parse_image_data((const uint8_t*)base_addr, 0x7FFFFFFF, img);
}

int image_load_from_bin(const char* bin_path, preprocessed_image_t* img) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open image file: %s\n", bin_path);
        return -1;
    }
    
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint8_t* buffer = (uint8_t*)malloc(file_size);
    if (!buffer) {
        fclose(f);
        return -1;
    }
    
    if (fread(buffer, 1, file_size, f) != file_size) {
        free(buffer);
        fclose(f);
        return -1;
    }
    fclose(f);
    
    int ret = parse_image_data(buffer, file_size, img);
    free(buffer); // 복사 완료했으므로 임시 버퍼 해제
    
    return ret;
}

void image_free(preprocessed_image_t* img) {
    if (!img) return;
    if (img->data) {
        free(img->data);
        img->data = NULL;
    }
}
