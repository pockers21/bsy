#pragma once


#include "bsy/common.hpp"
#include <cblas.h>
#include <math.h>

namespace bsy{

template<typename Dtype>
void partical_specialization(Dtype  value);



// C = alpha * A * B + beta * C
// shape :
//      A: (M,K)
//      B: (K,N)
//      C: (M,N)

template <typename Dtype>
void bsy_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void bsy_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const Dtype alpha, const Dtype* A, const Dtype* x,
    const Dtype beta, Dtype* y);

template <typename Dtype>
void bsy_cpu_vectors_simple_add(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void bsy_cpu_vector_scale_add(const int N, const Dtype alpha, const Dtype* X,
                            const Dtype beta, Dtype* Y);

template<typename Dtype>
void bsy_cpu_vector_add_scalar(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void bsy_scal(const int N, const Dtype alpha, Dtype *X);




#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(bsy/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void bsy_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void bsy_gpu_scal(const int N, const Dtype alpha, Dtype *X);

#ifndef CPU_ONLY
template <typename Dtype>
void bsy_gpu_scal(const int N, const Dtype alpha, Dtype* X, cudaStream_t str);
#endif

template <typename Dtype>
void bsy_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void bsy_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void bsy_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void bsy_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void bsy_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void bsy_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void bsy_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void bsy_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void bsy_gpu_sqrt(const int n, const Dtype* a, Dtype* y);



template <typename Dtype>
void bsy_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void bsy_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void bsy_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void bsy_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void bsy_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void bsy_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void bsy_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void bsy_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}