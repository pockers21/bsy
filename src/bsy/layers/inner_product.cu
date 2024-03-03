#include <vector>

#include "bsy/layers/inner_product_layer.hpp"

namespace bsy {

template <typename Dtype>
void InnerProductLayer<Dtype>::ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
    const vector<DataBlock<Dtype>*>& top) {


  const Dtype* bottom_data = bottom[0]->GetGpuData();
  Dtype* top_data = top[0]->GetGpuData();
  const Dtype* weight = this->data_blocks_[0]->GetGpuData();
  if (M_ == 1) {
    bsy_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      bsy_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->data_blocks_[1]->GetGpuData(), top_data);
  } else {

    bsy_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      bsy_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.GetGpuData(),
                            this->data_blocks_[1]->GetGpuData(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::BackwardGpu(const vector<DataBlock<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<DataBlock<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->GetGpuDiff();
    const Dtype* bottom_data = bottom[0]->GetGpuData();

    // Gradient with respect to weight
    if (transpose_) {
      bsy_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->data_blocks_[0]->GetGpuDiff());
    } else {
      bsy_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->data_blocks_[0]->GetGpuDiff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->GetGpuDiff();
    // Gradient with respect to bias
    bsy_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.GetGpuData(), (Dtype)1.,
        this->data_blocks_[1]->GetGpuDiff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->GetGpuDiff();

    const Dtype* top_cpu_diff = top[0]->GetCpuDiff();
    
    // Gradient with respect to bottom data
    if (transpose_) {
      bsy_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->data_blocks_[0]->GetGpuData(),
          (Dtype)0., bottom[0]->GetGpuDiff());
    } else {
      bsy_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->data_blocks_[0]->GetGpuData(),
         (Dtype)0., bottom[0]->GetGpuDiff());
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace bsy
