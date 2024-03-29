#include <algorithm>
#include <vector>

#include "bsy/layers/softmax_layer.hpp"
#include "bsy/util/math_functions.hpp"

namespace bsy {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top) {
  
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);

  Dtype* multiplier_data = sum_multiplier_.GetCpuData();
  bsy_set(sum_multiplier_.count(), Dtype(1), multiplier_data);

  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);

  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<DataBlock<Dtype>*>& bottom,
    const vector<DataBlock<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->GetCpuData();
  Dtype* top_data = top[0]->GetCpuData();
  Dtype* scale_data = scale_.GetCpuData();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  bsy_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    bsy_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    bsy_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.GetCpuData(), scale_data, 1., top_data);
    // exponentiation
    bsy_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    bsy_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.GetCpuData(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      bsy_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<DataBlock<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<DataBlock<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->GetCpuDiff();
  const Dtype* top_data = top[0]->GetCpuData();
  Dtype* bottom_diff = bottom[0]->GetCpuDiff();
  Dtype* scale_data = scale_.GetCpuData();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  bsy_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = bsy_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    bsy_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.GetCpuData(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  bsy_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace bsy
