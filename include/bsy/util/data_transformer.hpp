#pragma once

#include "bsy/data_block.hpp"
#include <cstdlib>
#include <ctime>

namespace bsy{

template<typename Dtype>
class DataTransformer {
    public:
        explicit DataTransformer(const TransformationParameter& param, Phase phase);
        virtual ~DataTransformer() {}

        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to the data.
        *
        * @param datum
        *    Datum containing the data to be transformed.
        * @param transformed_data block
        *    This is destination data block.
        */
        void Transform(const Datum& datum, DataBlock<Dtype>* transformed_data_block);

        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to a vector of Datum.
        *
        * @param datum_vector
        *    A vector of Datum containing the data to be transformed.
        * @param transformed_data block
        *    This is destination data block.
        */
        void Transform(const vector<Datum> & datum_vector,
                    DataBlock<Dtype>* transformed_data_block);

        /**
        * @brief Infers the shape of transformed_data block will have when
        *    the transformation is applied to the data.
        *
        * @param datum
        *    Datum containing the data to be transformed.
        */
        vector<int> InferDataBlockShape(const Datum& datum);

        /**
        * @brief Infers the shape of transformed_data block will have when
        *    the transformation is applied to the data.
        *    It uses the first element to infer the shape of the data block.
        *
        * @param datum_vector
        *    A vector of Datum containing the data to be transformed.
        */
        vector<int> InferDataBlockShape(const vector<Datum> & datum_vector);

    private:
        void Transform(const Datum& datum, Dtype* transformed_data);

        int Rand(const int upper_bound) const;
        TransformationParameter param_;

        Phase phase_;
        DataBlock<Dtype> data_mean_;
        vector<Dtype> mean_values_;

}