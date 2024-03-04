#pragma once

#include "bsy/util/db.hpp"
#include "bsy/data_block.hpp"
#include "bsy/layer.hpp"

namespace bsy{

template<typename Dtype>
class DataLayer: Layer<Dtype> {
    public:
        explicit DataLayer(const LayerParameter& param);

        virtual void LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top);

        virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
          const vector<DataBlock<Dtype>*>& top) {}

        virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top);

        virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top)；

        virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<DataBlock<Dtype>*>& bottom){};

        virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<DataBlock<Dtype>*>& bottom){};

        virtual void LoadBatch(Batch<Dtype>* batch) = 0;
    private:
        shared_ptr<DB> db_;
        DataParam data_param_;
        TransformationParameter transform_param_;
        shared_ptr<DataTransformer> transformer_;
        bool out_put_labels_;
};

template <typename Dtype>
class Batch {
 public:
  DataBlock<Dtype> data_, label_;
};




}