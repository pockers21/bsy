#pragma once

#include "bsy/util/db.hpp"
#include "bsy/data_block.hpp"
#include "bsy/layer.hpp"

namespace bsy{

template<typename Dtype>
class DataLayer: Layer<Dtype> {
    public:

    private:
        shared_ptr<DB> db_;
};
}