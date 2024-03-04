#include <vector>

#include "bsy/layers/inner_product_layer.hpp"

namespace bsy {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top) {
  
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();

 
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->data_blocks_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->data_blocks_.resize(2);
    } else {
      this->data_blocks_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->data_blocks_[0].reset(new DataBlock<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<DistributeGeneratorParameter<Dtype> > weight_filler(GetGenerator<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Generate(this->data_blocks_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->data_blocks_[1].reset(new DataBlock<Dtype>(bias_shape));
      shared_ptr<DistributeGeneratorParameter<Dtype> > bias_filler(GetGenerator<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Generate(this->data_blocks_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->data_blocks_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<DataBlock<Dtype>*>& bottom,
      const vector<DataBlock<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).


  vector<int> top_shape = bottom[0]->shape();
  
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;


  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    bsy_set(M_, Dtype(1), bias_multiplier_.GetCpuData());
  }


}

template <typename Dtype>
void InnerProductLayer<Dtype>::ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
    const vector<DataBlock<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->GetCpuData();
  Dtype* top_data = top[0]->GetCpuData();
  const Dtype* weight = this->data_blocks_[0]->GetCpuData();
  bsy_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    bsy_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.GetCpuData(),
        this->data_blocks_[1]->GetCpuData(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::BackwardCpu(const vector<DataBlock<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<DataBlock<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->GetCpuDiff();
    const Dtype* bottom_data = bottom[0]->GetCpuData();
    // Gradient with respect to weight
    if (transpose_) {
      bsy_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->data_blocks_[0]->GetCpuDiff());
    } else {
      bsy_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->data_blocks_[0]->GetCpuDiff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->GetCpuDiff();
    // Gradient with respect to bias
    bsy_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.GetCpuData(), (Dtype)1.,
        this->data_blocks_[1]->GetCpuDiff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->GetCpuDiff();
    // Gradient with respect to bottom data
    if (transpose_) {
      bsy_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->data_blocks_[0]->GetCpuData(),
          (Dtype)0., bottom[0]->GetCpuDiff());
    } else {
      bsy_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->data_blocks_[0]->GetCpuData(),
          (Dtype)0., bottom[0]->GetCpuDiff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace bsy
