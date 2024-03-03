#include "bsy/util/data_transformer.hpp"

namespace bsy{

template <typename Dtype>
DataTransformer::DataTransformer(const TransformationParameter& param,
    Phase phase):param_(param), phase_(phase) {
    // check if we want to use mean_value
    if (param_.mean_value_size() > 0) {
        for (int c = 0; c < param_.mean_value_size(); ++c) {
          mean_values_.push_back(param_.mean_value(c));
        }
    }
}

template <typename Dtype>
DataTransformer::Rand(const int upper_bound) {
    std::srand(std::time(0));

    int random_number = std::rand() % upper_bound;
    return random_number;
}

template <typename Dtype>
void DataTransformer::Transform(const Datum& datum, Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  //const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_file = false;
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int transformed_data_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = c * datum_height * datum_width + (h_off + h) * datum_width + (w_off + w)
        if (do_mirror) {
          transformed_data_index = c * height * weight + h * width + (width - 1 - w)
        } else {
          transformed_data_index = c * height * weight + h * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[transformed_data_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[transformed_data_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[transformed_data_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template <typename Dtype>
void DataTransformer::Transform(const Datum& datum, DataBlock<Dtype>* transformed_data_block) {
    const int crop_size = param_.crop_size();
    const int datum_channels = datum.channels();
    const int datum_height = datum.height();
    const int datum_width = datum.width();

    // Check dimensions.
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, datum_channels);
    CHECK_LE(height, datum_height);
    CHECK_LE(width, datum_width);
    CHECK_GE(num, 1);

    if (crop_size) {
        CHECK_EQ(crop_size, height);
        CHECK_EQ(crop_size, width);
    } else {
        CHECK_EQ(datum_height, height);
        CHECK_EQ(datum_width, width);
    }

    Dtype* transformed_data = transformed_blob->GetCpuData();
    Transform(datum, transformed_data);
}

template <typename Dtype>
void DataTransformer::Transform(const vector<Datum> & datum_vector,
                    DataBlock<Dtype>* transformed_data_block) {
    const int datum_num = datum_vector.size();
    const int num = transformed_data_block->num();
    const int channels = transformed_data_block->channels();
    const int height = transformed_data_block->height();
    const int width = transformed_data_block->width();

    CHECK_GT(datum_num, 0) << "There is no datum to add";
    CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_data_block->num()";
    Blob<Dtype> uni_blob(1, channels, height, width);
    for (int item_id = 0; item_id < datum_num; ++item_id) {
        int offset = transformed_data_block->offset(item_id);
        uni_blob.set_cpu_data(transformed_data_block->GetCpuData() + offset);
        Transform(datum_vector[item_id], &uni_blob);
    }
}

template <typename Dtype>
void DataTransformer::InferDataBlockShape(const Datum& datum) {
    const int crop_size = param_.crop_size();
    const int datum_channels = datum.channels();
    const int datum_height = datum.height();
    const int datum_width = datum.width();
    // Check dimensions.
    CHECK_GT(datum_channels, 0);
    CHECK_GE(datum_height, crop_size);
    CHECK_GE(datum_width, crop_size);
    // Build BlobShape.
    vector<int> shape(4);
    shape[0] = 1;
    shape[1] = datum_channels;
    shape[2] = (crop_size)? crop_size: datum_height;
    shape[3] = (crop_size)? crop_size: datum_width;
    return shape;
}

template <typename Dtype>
void DataTransformer::InferDataBlockShape(const vector<Datum> & datum_vector) {
    const int num = datum_vector.size();
    CHECK_GT(num, 0) << "There is no datum to in the vector";
    // Use first datum in the vector to InferBlobShape.
    vector<int> shape = InferBlobShape(datum_vector[0]);
    // Adjust num to the size of the vector.
    shape[0] = num;
    return shape;
}
}