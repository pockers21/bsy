#include "bsy/layer/data_layer.hpp"

namespace bsy {

template <typename Dtype>
void DataLayer::ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) {
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    Datum datum;

    datum.ParseFromString(db_->GetCurrentValue());

    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->transformer_->InferDataBlockShape(datum);
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);


    // label
    if (this->output_labels_) {
        vector<int> label_shape(1, batch_size);
        top[1]->Reshape(label_shape);
    }

    // Reshape to loaded data.
    top[0]->SetGpuData(batch->data_.SetGpuData());
    if (this->output_labels_) {
        // Reshape to loaded labels.
        top[1]->SetGpuData(batch->label_.SetGpuData());
    }

            }
}