#include <algorithm>
#include <vector>

#include "bsy/layers/relu_layer.hpp"

namespace bsy {

template <typename Dtype>
void ReLULayer<Dtype>::ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
    const vector<DataBlock<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->GetCpuData();
  Dtype* top_data = top[0]->GetCpuData();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::BackwardCpu(const vector<DataBlock<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<DataBlock<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->GetCpuData();
    const Dtype* top_diff = top[0]->GetCpuDiff();
    Dtype* bottom_diff = bottom[0]->GetCpuDiff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
