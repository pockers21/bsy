#pragma once

#include <vector>

#include "bsy/data_block.hpp"
#include "bsy/layer.hpp"
#include "bsy/proto/bsy.pb.h"

namespace bsy {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmax"; }


 protected:
  virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);
  virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  DataBlock<Dtype> sum_multiplier_;
  /// scale is an intermediate DataBlock to hold temporary results.
  DataBlock<Dtype> scale_;
};

}  // namespace bsy
