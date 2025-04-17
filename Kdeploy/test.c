#include <immintrin.h> // Required for AVX/FMA intrinsics
#include <stdio.h>

void fused_multiply_add(const float* A, const float* B, const float* C, float* result) {
  // Load 8 float32 values into AVX registers
  __m256 a = _mm256_loadu_ps(A);
  __m256 b = _mm256_loadu_ps(B);
  __m256 c = _mm256_loadu_ps(C);

  // Perform the FMA: (a * b) + c
  __m256 r = _mm256_fmadd_ps(a, b, c);

  // Store result back to memory
  _mm256_storeu_ps(result, r);
}

int main() {
  float A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float B[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  float C[8] = {100, 200, 300, 400, 500, 600, 700, 800};
  float result[8];

  fused_multiply_add(A, B, C, result);

  // Print the result
  for (int i = 0; i < 8; ++i) {
    printf("result[%d] = %.2f\n", i, result[i]);
  }

  return 0;
}
