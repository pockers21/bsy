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
        data_.reset(new MemoryBlock(capacity_ * sizeof(Dtype)));
        diff_.reset(new MemoryBlock(capacity_ * sizeof(Dtype)));
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
        data_.reset(new MemoryBlock(size));
        diff_.reset(new MemoryBlock(size));
    }
    data_->set_cpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetCpuDiff(Dtype* data) {
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock(size));
        diff_.reset(new MemoryBlock(size));
    }
    diff_->set_cpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetGpuData(Dtype* data) {
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock(size));
        diff_.reset(new MemoryBlock(size));
    }
    this->diff_->set_gpu_data(data);
}

template <typename Dtype>
void DataBlock<Dtype>::SetGpuDiff(Dtype* data) {

    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size) {
        data_.reset(new MemoryBlock(size));
        diff_.reset(new MemoryBlock(size));
    }
    this->diff_->set_gpu_data(data);
}


template <typename Dtype>
void  DataBlock<Dtype>::ShareData(const DataBlock& other) {
    CHECK_EQUAL(count_, other.GetCount());
    this->data_ = other.GetDataMemBlock();
}

template <typename Dtype>
void DataBlock<Dtype>::ShareDiff(const DataBlock& other) {
    CHECK_EQUAL(this->count_, other.GetCount());
    this->diff_ = other.GetDiffMemBlock();
}







}