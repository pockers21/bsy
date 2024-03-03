#pragma once

#include <vector>

#include "bsy/memory_block.hpp"
#include "bsy/layer.hpp"
#include "bsy/proto/bsy.pb.h"

namespace bsy {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  

 protected:
  virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);
  virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  PoolingParameter_RoundMode round_mode_;
  DataBlock<Dtype> rand_idx_;
  DataBlock<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
