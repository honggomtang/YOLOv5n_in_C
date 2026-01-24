#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIMS 8

// 텐서 정보 구조체
typedef struct {
    char* name;              // 텐서 이름 (동적 할당)
    float* data;             // 텐서 데이터 (동적 할당, 4byte 정렬 보장)
    int32_t ndim;            // 차원 수
    int32_t shape[MAX_TENSOR_DIMS]; // shape 배열 (고정 크기, 정렬 문제 방지)
    size_t num_elements;     // 총 원소 개수
} tensor_info_t;

// 가중치 로더 구조체
typedef struct {
    tensor_info_t* tensors;
    int32_t num_tensors;
} weights_loader_t;

// ===== 임베디드용: 메모리 직접 접근 (파일 시스템 없음) =====
// base_addr: .bin 데이터가 로드된 메모리 시작 주소
int weights_init_from_memory(uintptr_t base_addr, weights_loader_t* loader);

// ===== 개발/테스트용: 파일 시스템에서 로드 =====
// 반환값: 0 성공, -1 실패
int weights_load_from_file(const char* bin_path, weights_loader_t* loader);

// 특정 이름의 텐서 찾기
// 반환값: 텐서 포인터, 없으면 NULL
const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name);

// 특정 이름의 텐서 데이터 포인터 가져오기 (편의 함수)
// 반환값: 텐서 데이터 포인터, 없으면 NULL
const float* weights_get_tensor_data(const weights_loader_t* loader, const char* name);

// 가중치 로더 해제
void weights_free(weights_loader_t* loader);

#endif // WEIGHTS_LOADER_H
