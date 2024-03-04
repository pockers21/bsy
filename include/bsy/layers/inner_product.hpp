#pragma once

#include <vector>

#include "bsy/layer.hpp"
#include "bsy/proto/caffe.pb.h"
#include "bsy/common.hpp"

namespace bsy {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }


 protected:
  virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);
  virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  DataBlock<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace bsy

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
