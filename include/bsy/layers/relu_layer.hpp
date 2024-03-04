#pragma once

#include <vector>

#include "bsy/common.hpp"
#include "bsy/layer.hpp"

namespace bsy {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class ReLULayer : {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReLULayer(const LayerParameter& param) {}

  virtual inline const char* type() const { return "ReLU"; }

  virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
          const vector<DataBlock<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
  };

 protected:
  /**
   * @param bottom input DataBlock vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output DataBlock vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = \max(0, x)
   *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed outputs are @f$ y = \max(0, x) + \nu \min(0, x) @f$.
   */
  virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);
  virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the ReLU inputs.
   *
   * @param top output DataBlock vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input DataBlock vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            0 & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$ if propagate_down[0], by default.
   *      If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed gradients are @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            \nu \frac{\partial E}{\partial y} & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);
  virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom);
};

}  // namespace bsy

#endif  // bsy_RELU_LAYER_HPP_
