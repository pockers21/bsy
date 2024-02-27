#pragma once
#include <random>
#include <cmath>

#include "bsy/common.hpp"

namespace bsy{


template<typename Dtype>
class Generator {
public:
    explicit Generator(const DistributeGeneratorParameter& param){
        param_ = param;
        e_.reset(new std::default_random_engine(Get_Seed()));
    };
    virtual  ~Generator(){}
    virtual void Generate(Dtype * arr_out, const int length)=0;

    // random seeding
    unsigned Get_Seed(void) {
        std::random_device rd;
        unsigned seed = rd();

        if (seed == 0) {
            seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
            LOG(INFO) << "Random seed generated from timestamp: " << seed;
        } else {
            LOG(INFO) << "Random seed generated from random_device: " << seed;
        }
        return seed;

    }
protected:
    shared_ptr<std::default_random_engine> e_;
    DistributeGeneratorParameter param_;
};

template<typename Dtype>
class ConstGenerator: public Generator<Dtype> {

public:
    explicit ConstGenerator(const DistributeGeneratorParameter& param)
          :Generator<Dtype>(param){}

    virtual void Generate(Dtype * arr_out, const int length){
        const Dtype  value = this->param_.constant();
        for(int i = 0; i < length; i++){
            arr_out[i] = value;
        }
    }

};

template<typename Dtype>
class UniformGenerator: public Generator<Dtype> {

public:
    explicit UniformGenerator(const DistributeGeneratorParameter& param)
          :Generator<Dtype>(param){}

    virtual void Generate(Dtype * arr_out, const int length){
        std::uniform_real_distribution<Dtype> uniform_generator(this->param_.min(), this->param_.max());
        for(int i = 0; i < length; i++){
            arr_out[i] = uniform_generator(*(this->e_));
        }
    }
};

template<typename Dtype>
class GaussianGenerator: public Generator<Dtype> {
public:
    explicit GaussianGenerator(const DistributeGeneratorParameter& param)
          :Generator<Dtype>(param){}

    virtual void Generate(Dtype * arr_out, const int length){
        std::normal_distribution<Dtype> gaussian_generator(this->param_.min(), this->param_.max());
        for(int i = 0; i < length; i++){
            arr_out[i] = gaussian_generator(*(this->e_));
        }
        int sparse = this->param_.sparse();
        CHECK_GE(sparse, -1);
        if(sparse >= 0){

            Dtype probability = Dtype(sparse)/Dtype(length);
            Dtype * mask = new Dtype(length);

            std::bernoulli_distribution bernoulli_generator(probability);
            for(int i = 0; i < length; i++){
                mask[i] =  bernoulli_generator(*(this->e_));
                arr_out[i] *= mask[i];
            }
            delete []mask;
        }
    }
};


}

