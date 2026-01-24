#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 헬퍼: 메모리에서 안전하게 값 읽기 (Unaligned access 방지용 memcpy 사용)
static inline void safe_read(void* dest, const uint8_t** src, size_t size) {
    memcpy(dest, *src, size);
    *src += size;
}

// 공통 파싱 로직
static int parse_weights_data(const uint8_t* ptr, size_t data_len, weights_loader_t* loader) {
    const uint8_t* curr = ptr;
    const uint8_t* end = ptr + data_len;

    // 1. 텐서 개수 읽기
    if (curr + 4 > end) return -1;
    uint32_t num_tensors;
    safe_read(&num_tensors, &curr, 4);

    loader->num_tensors = (int32_t)num_tensors;
    loader->tensors = (tensor_info_t*)calloc(num_tensors, sizeof(tensor_info_t));
    if (!loader->tensors) return -1;

    for (int i = 0; i < (int)num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];

        // 2. 키 이름 길이 읽기
        if (curr + 4 > end) return -1;
        uint32_t key_len;
        safe_read(&key_len, &curr, 4);

        if (key_len > 1024) return -1; // Sanity check
        if (curr + key_len > end) return -1;

        // 3. 키 이름 할당 및 복사
        t->name = (char*)malloc(key_len + 1);
        if (!t->name) return -1;
        safe_read(t->name, &curr, key_len);
        t->name[key_len] = '\0';

        // 4. 차원 수 읽기
        if (curr + 4 > end) return -1;
        uint32_t ndim;
        safe_read(&ndim, &curr, 4);
        t->ndim = (int32_t)ndim;

        if (ndim > MAX_TENSOR_DIMS) return -1;

        // 5. Shape 읽기 (배열에 값 복사)
        if (curr + ndim * 4 > end) return -1;
        t->num_elements = 1;
        for (int j = 0; j < (int)ndim; j++) {
            uint32_t dim_val;
            safe_read(&dim_val, &curr, 4);
            t->shape[j] = (int32_t)dim_val;
            t->num_elements *= dim_val;
        }

        // 6. 데이터 복사 (malloc으로 정렬된 메모리 할당)
        size_t data_bytes = t->num_elements * sizeof(float);
        if (curr + data_bytes > end) return -1;

        t->data = (float*)malloc(data_bytes);
        if (!t->data) return -1;
        
        // 데이터 memcpy (Unaligned source -> Aligned dest)
        safe_read(t->data, &curr, data_bytes);
    }

    return 0;
}

int weights_init_from_memory(uintptr_t base_addr, weights_loader_t* loader) {
    return parse_weights_data((const uint8_t*)base_addr, 0x7FFFFFFF, loader);
}

int weights_load_from_file(const char* bin_path, weights_loader_t* loader) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", bin_path);
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

    int ret = parse_weights_data(buffer, file_size, loader);
    free(buffer);
    
    if (ret != 0) {
        weights_free(loader);
    }
    return ret;
}

const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name) {
    char search_name[512];
    
    // 1. 정확한 매치
    for (int i = 0; i < loader->num_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            return &loader->tensors[i];
        }
    }

    // 2. model.* -> model.model.model.* 매핑 시도
    if (strncmp(name, "model.", 6) == 0) {
        snprintf(search_name, sizeof(search_name), "model.model.%s", name);
        for (int i = 0; i < loader->num_tensors; i++) {
            if (strcmp(loader->tensors[i].name, search_name) == 0) {
                return &loader->tensors[i];
            }
        }
    }

    return NULL;
}

const float* weights_get_tensor_data(const weights_loader_t* loader, const char* name) {
    const tensor_info_t* t = weights_find_tensor(loader, name);
    if (!t) {
        fprintf(stderr, "Warning: Weight not found: %s\n", name);
        return NULL;
    }
    return t->data;
}

void weights_free(weights_loader_t* loader) {
    if (!loader || !loader->tensors) return;

    for (int i = 0; i < loader->num_tensors; i++) {
        if (loader->tensors[i].name) free(loader->tensors[i].name);
        if (loader->tensors[i].data) free(loader->tensors[i].data);
    }
    free(loader->tensors);
    loader->tensors = NULL;
    loader->num_tensors = 0;
}
