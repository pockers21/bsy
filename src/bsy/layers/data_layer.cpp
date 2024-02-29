#include "bsy/layer/data_layer.hpp"

namespace bsy {

template<typename Dtype>
DataLayer::DataLayer(const LayerParameter& param) {
    data_param_ = param.
    transform_param_ = param.transform_param();
    transformer_.reset(new DataTransformer(transform_param_));
    db_.reset(db::GetDB(param.backend()));
    db_->Open(param.data_param().source(), db::READ);
}

template<typename Dtype>
virtual void LoadBatch(Batch<Dtype>* batch) {
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    const int batch_size = this->layer_param_.data_param().batch_size();
    Datum datum;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        datum.ParseFromString(db->GetCurrentValue());
        if (item_id == 0) {
            // Reshape according to the first datum of each batch
            // on single input batches allows for inputs of varying dimension.
            // Use data_transformer to infer the expected data block shape from datum.
            vector<int> top_shape = this->transformer_->InferDataBlockShape(datum);
            this->transformed_data_.Reshape(top_shape);
            // Reshape batch according to the batch_size.
            top_shape[0] = batch_size;
            batch->data_.Reshape(top_shape);
        }

        int offset = batch->data_.offset(item_id);
        Dtype* top_data = batch->data_.GetCpuData();
        this->transformed_data_.SetCpuData(top_data + offset);
        this->transformer_->Transform(datum, &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_) {
          Dtype* top_label = batch->label_.GetCpuData();
          top_label[item_id] = datum.label();
        }
        db_->CursorNext();
    }



}

template<typename Dtype>
DataLayer::LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) {
    if (top.size() == 1) {
       output_labels_ = false;
    } else {
        output_labels_ = true;
    }
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
}


template<typename Dtype>
DataLayer::ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) {

    // Reshape to loaded data.
    top[0]->ReshapeLike(prefetch_current_->data_);
    top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
    if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
    }
}

template<typename Dtype>
DataLayer::BackwardCpu(const LayerParameter& param) {}

}