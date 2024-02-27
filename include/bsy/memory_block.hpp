#pragma once
#include "bsy/common.hpp"

namespace bsy{

template <typename Dtype>
class MemoryBlock{
    public:
        MemoryBlock();
        MemoryBlock(size_t size);
        enum CurrentStatus {HEAD_UNITED, HEAD_AT_CPU, HEAD_AT_GPU};
        const CurrentStatus GetStatus() const {return  status_;}


        void SetCpuData(const Dtype * data);
        void *GetCpuData(const Dtype * data) const;

        void SetGpuData();
        void *GetGpuData() const;

    private:

        void * cpu_ptr_;
        void * gpu_ptr_;
        size_t size_;
        CurrentStatus status_;
        bool cpu_mem_occupied_;
        bool gpu_mem_occupied_;
        FORBID_COPY_AND_ASSIGN(MemoryBlock);
};

}