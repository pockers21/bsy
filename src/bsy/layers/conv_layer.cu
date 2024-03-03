#include <vector>

#include "bsy/layers/conv_layer.hpp"

namespace bsy {

template <typename Dtype>
void ConvolutionLayer<Dtype>::ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top) {
  const Dtype* weight = this->data_blocks_[0]->GetGpuData();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->GetGpuData();
    Dtype* top_data = top[i]->GetGpuData();
    for (int n = 0; n < this->num_; ++n) {
      this->ForwardGpuGemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->data_blocks_[1]->GetGpuData();
        this->ForwardGpuBias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::BackwardGpu(const vector<DataBlock<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<DataBlock<Dtype>*>& bottom) {
  const Dtype* weight = this->data_blocks_[0]->GetGpuData();
  Dtype* weight_diff = this->data_blocks_[0]->GetGpuDiff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->GetGpuDiff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->data_blocks_[1]->GetGpuDiff();
      for (int n = 0; n < this->num_; ++n) {
        this->BackwardGpuBias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->GetGpuData();
      Dtype* bottom_diff = bottom[i]->GetGpuDiff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->WeightGpuGemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->BackwardGpuGemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
