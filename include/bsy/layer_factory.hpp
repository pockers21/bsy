#pragma once
#include <functional>
#include "bsy/common.hpp"

namespace bsy{

template<typename Dtype>

class LayerFactory {
    using generate_function = std::function<shared_ptr<Layer<Dtype>> Layer (const LayerParam&)>;

    public:

        static LayerFactory&  GetInstance() {
            static LayerFactory singleton_ptr_;
            //singleton_ptr_.reset(new LayerFactory());
            return singleton_ptr_
        }

        void AddClassToRegister(const string& type, const generate_function& regist_function) {
            CHECK_EQUAL(global_register_map_.count(type),0) << "Layer type " << type << " already registered.";
            global_register_map_[type] = regist_function;
        }

    private:
        LayerFactory(){}
        static LayerFactory  singleton_ptr_;
        //global regist map, used for generate specific
        map<string, generate_function> global_register_map_;
};


REGITSER_CLASS(CLASS_NAME) \
    template<typename Dtype> \
    shared_ptr<Layer<Dtype>> Create##CLASS_NAME(const LayerParam* param) { \
        return shared_ptr<Layer>(new Layer(param)); \
    } \

    LayerFactory::GetInstance().AddClassToRegister(#type, Create##CLASS_NAME<float>); \
    LayerFactory::GetInstance().AddClassToRegister(#type, Create##CLASS_NAME<double>); \
}