#include <math_functions.h>  // CUDA's, not bsy's, for fabs, signbit
#include <cmath>
#include "bsy/common.hpp"
#include "bsy/util/math.hpp"

namespace bsy {

template <>
voidbsy_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Bsy::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
voidbsy_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Bsy::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
voidbsy_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Bsy::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
voidbsy_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Bsy::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
voidbsy_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Bsy::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
voidbsy_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Bsy::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

voidbsy_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
  }
}

template <>
voidbsy_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Bsy::cublas_handle(), N, &alpha, X, 1));
}

template <>
voidbsy_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Bsy::cublas_handle(), N, &alpha, X, 1));
}

template <>
voidbsy_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Bsy::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Bsy::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Bsy::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Bsy::cublas_handle(), initial_stream));
}

template <>
voidbsy_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Bsy::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Bsy::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Bsy::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Bsy::cublas_handle(), initial_stream));
}

template <>
voidbsy_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
 bsy_gpu_scal<float>(N, beta, Y);
 bsy_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
voidbsy_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
 bsy_gpu_scal<double>(N, beta, Y);
 bsy_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
voidbsy_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Bsy::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
voidbsy_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Bsy::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
voidbsy_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Bsy::cublas_handle(), n, x, 1, y));
}

template <>
voidbsy_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Bsy::cublas_handle(), n, x, 1, y));
}

template <>
voidbsy_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Bsy::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Bsy::cublas_handle(), n, &alpha, y, 1));
}

template <>
voidbsy_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Bsy::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Bsy::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
voidbsy_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template voidbsy_gpu_set<int>(const int N, const int alpha, int* Y);
template voidbsy_gpu_set<float>(const int N, const float alpha, float* Y);
template voidbsy_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
voidbsy_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
voidbsy_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
voidbsy_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
voidbsy_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
voidbsy_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
voidbsy_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
voidbsy_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
voidbsy_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
voidbsy_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
voidbsy_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
voidbsy_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
voidbsy_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
voidbsy_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
voidbsy_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
voidbsy_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
voidbsy_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
voidbsy_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
voidbsy_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
voidbsy_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
voidbsy_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<BSY_GET_BLOCKS(N),BSY_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

voidbsy_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Bsy::curand_generator(), r, n));
}

template <>
voidbsy_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Bsy::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
   bsy_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
   bsy_gpu_add_scalar(n, a, r);
  }
}

template <>
voidbsy_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Bsy::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
   bsy_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
   bsy_gpu_add_scalar(n, a, r);
  }
}

template <>
voidbsy_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Bsy::curand_generator(), r, n, mu, sigma));
}

template <>
voidbsy_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Bsy::curand_generator(), r, n, mu, sigma));
}

}  // namespacebsy
