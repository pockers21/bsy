#include "bsy/data_block.hpp"

namespace bsy{

template <typename Dtype>
DataBlock<Dtype>::DataBlock():count_(0),capacity_(0),data_(), diff_(){}


template <typename Dtype>
DataBlock<Dtype>::DataBlock(const int num, const int channal, const int height, const int width){
    count_ = 0;
    capacity_ = 0;
    Reshape(num, channal, height, width);
}

template <typename Dtype>
DataBlock<Dtype>::DataBlock(const vector<int> & shape){
    count_ = 0;
    capacity_ = 0;
    Reshape(shape);
}

template <typename Dtype>
void DataBlock<Dtype>::Reshape(const int num, const int channal, const int height, const int width){
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channal;
    shape[2] = height;
    shape[3] = width;
    Reshape(shape);
}

template <typename Dtype>
void DataBlock<Dtype>::Reshape(const DataBlock& other) {
    Reshape(other.shape());
}


template <typename Dtype>
void DataBlock<Dtype>::Reshape(const vector<int> shape) {
    count_ = 1;
    shape_.resize(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
        CHECK_GE(shape[i], 0);

        count_ *= shape[i];
        shape_[i] = shape[i];
    }

    if (count_ > capacity_) {
        capacity_ = count_;
        data_.reset(new MemoryBlock<Dtype>(capacity_ * sizeof(Dtype)));
        diff_.reset(new MemoryBlock<Dtype>(capacity_ * sizeof(Dtype)));
    }
}


template <typename Dtype>
Dtype * DataBlock<Dtype>::GetCpuData() const {
    CHECK(data_);
    return data_->GetCpuData();
}

template <typename Dtype>
Dtype * DataBlock<Dtype>::GetCpuDiff() const{
    CHECK(data_);
    return diff_->GetCpuData();

}

template <typename Dtype>
Dtype * DataBlock<Dtype>::GetGpuData() const {
    CHECK(data_);
    return data_->GetGpuData();
}

template <typename Dtype>
Dtype * DataBlock<Dtype>::GetGpuDiff() const {
    CHECK(data_);
    return diff_->GetGpuData();
}


template <typename Dtype>
void DataBlock<Dtype>::SetCpuData(Dtype* data) {

    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock<Dtype>(size));
        diff_.reset(new MemoryBlock<Dtype>(size));
    }
    data_->set_cpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetCpuDiff(Dtype* data) {
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock<Dtype>(size));
        diff_.reset(new MemoryBlock<Dtype>(size));
    }
    diff_->set_cpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetGpuData(Dtype* data) {
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock<Dtype>(size));
        diff_.reset(new MemoryBlock<Dtype>(size));
    }
    this->diff_->set_gpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetGpuDiff(Dtype* data) {

    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock<Dtype>(size));
        diff_.reset(new MemoryBlock<Dtype>(size));
    }
    this->diff_->set_gpu_data(data);
}


template <typename Dtype>
void  DataBlock<Dtype>::ShareData(const DataBlock& other) {
    CHECK_EQUAL(count_, other.GetCount());
    data_ = other.GetDataMemBlock();
}

template <typename Dtype>
void DataBlock<Dtype>::ShareDiff(const DataBlock& other) {
    CHECK_EQUAL(this->count_, other.GetCount());
    diff_ = other.GetDiffMemBlock();
}

template <typename Dtype>
Dtype Blob<Dtype>::AbsSumData() const {
    CHECK(data_);
    switch (data_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return 0;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_cpu_asum(count_, GetCpuData());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                Dtype asum;
                bsy_gpu_asum(count_, GetGpuData(), &asum);
                return asum;
              }
            #else
         break;


    }
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::AbsSumDiff() const {
    CHECK(data_);
    switch (diff_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return 0;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_gpu_asum(count_, GetCpuDiff());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                Dtype asum;
                bsy_gpu_asum(count_, GetGpuDiff(), &asum);
                return asum;
              }
            #else
         break;
    }
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::DotSumData() const {
    CHECK(data_);
    switch (data_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return 0;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_cpu_dot(count_, GetCpuData());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                Dtype dot;
                bsy_gpu_dot(count_, GetGpuData(), &dot);
                return dot;
              }
            #else
         break;


    }
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::DotSumDiff() const {
    CHECK(data_);
    switch (diff_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return 0;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_cpu_dot(count_, GetCpuDiff());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                Dtype dot;
                bsy_gpu_dot(count_, GetGpuDiff(), &dot);
                return dot;
              }
            #else
         break;
    }
    return 0;
}


template <typename Dtype>
Dtype Blob<Dtype>::ScaleData() const {
    CHECK(data_);
    switch (data_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return ;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_cpu_scale(count_, GetCpuData());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                Dtype scale;
                bsy_gpu_scale(count_, GetGpuData(), &dot);
                return ;
              }
            #else
         break;


    }
    return ;
}

template <typename Dtype>
Dtype Blob<Dtype>::ScaleDiff() const {
    CHECK(data_);
    switch (diff_->GetStatus()) {
        case MemoryBlock::HEAD_UNITED:
            return ;
        case MemoryBlock::HEAD_AT_CPU:
            return bsy_cpu_scale(count_, GetCpuDiff());
            break;
        case MemoryBlock::HEAD_AT_GPU:
            #ifndef CPU_ONLY
              {
                bsy_gpu_scale(count_, GetGpuDiff(), &dot);
                return ;
              }
            #else
         break;
    }
    return ;
}


}