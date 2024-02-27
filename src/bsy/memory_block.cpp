#include  "bsy/memory_block.hpp"

namespace bsy {


template<typename Dtype>
MemoryBlock::MemoryBlock()
       :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(HEAD_UNITED){}


template<typename Dtype>
MemoryBlock::MemoryBlock(const size_t size)
       :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(HEAD_UNITED){
    this->size_ = size;
}

template<typename Dtype>
MemoryBlock::~MemoryBlock(const size_t size){
    if(cpu_ptr_){
        free(cpu_ptr_);
    }
#ifdef CPU_ONLY
    if(gpu_ptr_){
        cudaFree(gpu_ptr_);
    }
#endif
}


template<typename Dtype>
void MemoryBlock::SetCpuData(Dtype * data){
    CHECK(data);
    if(this->cpu_mem_occupied_){
        free(this->cpu_ptr_);
        cpu_mem_occupied_ = false;
    }
    cpu_ptr_ = data;
    status_ = HEAD_AT_CPU;
}


template<typename Dtype>
void* GetCpuData(){
    switch{status_}{
        case HEAD_UNITED:
            cpu_ptr_ = malloc(size_);
            CHECK(*cpu_ptr_) << "host allocation of size " << size_ << " failed";

            cpu_mem_occupied_ = true;
            break;
        case HEAD_AT_CPU:
            break;
        case HEAD_AT_GPU:
            CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault));
            cpu_mem_occupied_ = true;

    }
    status_ = HEAD_AT_CPU;
}


template<typename Dtype>
void MemoryBlock::SetGpuData(Dtype * data){
#ifndef CPU_ONLY
    CHECK(data);
    if(this->gpu_mem_occupied_){
        CUDA_CHECK(cudaFree(this->cpu_ptr_));
        gpu_mem_occupied_ = false;
    }
    gpu_ptr_ = data;
    status_ = HEAD_AT_GPU;
#endif
}


template<typename Dtype>
void* GetGpuData(){
#ifdef CPU_ONLY
    switch{status_}{
        case HEAD_UNITED:
            gpu_ptr_ = cudaMalloc(size_);
            CUDA_CHECK(cudaMemset(size_, 0, gpu_ptr_));
            CHECK(*gpu_ptr_) << "device allocation of size " << size_ << " failed";
            gpu_mem_occupied_ = true;
            break;
        case HEAD_AT_GPU:
            break;
        case HEAD_AT_CPU:
            CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault));
            gpu_mem_occupied_ = true;

    }
    status_ = HEAD_AT_GPU;

    return gpu_ptr_;

#endif
    return NULL;

}
}