
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types

// CUDA: use 512 threads per block
const int BSY_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int BSY_GET_BLOCKS(const int N) {
  return (N + BSY_CUDA_NUM_THREADS - 1) / BSY_CUDA_NUM_THREADS;
}

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

