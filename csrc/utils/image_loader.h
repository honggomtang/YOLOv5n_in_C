#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdint.h>

// 전처리된 이미지 정보
typedef struct {
    float* data;         // 이미지 데이터 (C, H, W) - NCHW 형식 (동적 할당, 정렬됨)
    int32_t c, h, w;     // 채널, 높이, 너비
    int32_t original_w, original_h;  // 원본 이미지 크기
    float scale;         // 리사이즈 스케일
    int32_t pad_x, pad_y;  // 패딩 위치
} preprocessed_image_t;

// ===== 임베디드용: 메모리 직접 접근 (파일 시스템 없음) =====
// base_addr: 이미지 데이터가 로드된 메모리 시작 주소
// 반환값: 0 성공, -1 실패
int image_init_from_memory(uintptr_t base_addr, preprocessed_image_t* img);

// ===== 개발/테스트용: 파일 시스템에서 로드 =====
// 반환값: 0 성공, -1 실패
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img);

// 이미지 해제
void image_free(preprocessed_image_t* img);

#endif // IMAGE_LOADER_H
