#pragma once


#include "bsy/common.hpp"
#include <cblas.h>

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


}