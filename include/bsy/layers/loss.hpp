#pragma once

#include <vector>

#include "bsy/data_block.hpp"
#include "bsy/layer.hpp"
#include "bsy/proto/bsy.pb.h"

namespace bsy {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two DataBlock%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton DataBlock representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<DataBlock<Dtype>*>& bottom, const vector<DataBlock<Dtype>*>& top);
  virtual void Reshape(
      const vector<DataBlock<Dtype>*>& bottom, const vector<DataBlock<Dtype>*>& top);

  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace bsy

#endif
