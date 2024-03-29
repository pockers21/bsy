
#include "bsy/util/math.hpp"


namespace bsy {

template<>
void partical_specialization<float>(float value )
{
  cout << "partical_specialization_float " << endl;
}


template<>
void partical_specialization<double>(double value)
{
  cout << "partical_specialization_double " << endl;
}

template<>
void bsy_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void bsy_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}



template <>
void bsy_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void bsy_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}



template <>
void bsy_cpu_vectors_simple_add<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void bsy_cpu_vectors_simple_add<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }


inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}


template <>
void bsy_cpu_scale<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void bsy_cpu_scale<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}


template <>
void bsy_cpu_vector_scale_add<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void bsy_cpu_vector_scale_add<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void bsy_cpu_vector_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template<>
void bsy_cpu_vector_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void bsy_cpu_mem_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void bsy_cpu_mem_set<int>(const int N, const int alpha, int* Y);
template void bsy_cpu_mem_set<float>(const int N, const float alpha, float* Y);
template void bsy_cpu_mem_set<double>(const int N, const double alpha, double* Y);





template <>
float bsy_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double bsy_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype bsy_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return bsy_cpu_strided_dot(n, x, 1, y, 1);
}

template
float bsy_cpu_dot<float>(const int n, const float* x, const float* y);

template
double bsy_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float bsy_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double bsy_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void bsy_cpu_scale_to_other<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void bsy_cpu_scale_to_other<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);

  cblas_dscal(n, alpha, y, 1);

}


template <typename Dtype>
void bsy_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if(!ProgramCpuMode) {
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));

    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
}


template void bsy_copy<int>(const int N, const int* X, int* Y);
template void bsy_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void bsy_copy<float>(const int N, const float* X, float* Y);
template void bsy_copy<double>(const int N, const double* X, double* Y);




}


