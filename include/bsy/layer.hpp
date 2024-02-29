#pragma once
#include "bsy/common.hpp"
#include "bsy/data_block.hpp"

namespace bsy{

template <typename Dtype>

template<Dtype>
class Layer{
    public：
        /**
           * You should not implement your own constructor. Any set up code should go
           * to SetUp(), where the dimensions of the bottom blobs are provided to the
           * layer.
           */
        explicit Layer(const LayerParameter param):layer_param_(param) {
            phase_ = param.phase();
            if(param.data_blocks_size() > 0) {
                data_blocks_.resize(param.data_blocks_size());
                for(int i=0; i < param.data_blocks_size(); i++) {
                    data_blocks_[i].reset(new DataBlock<Dtype>());
                    data_blocks_[i]->FromProto(param.data_blocks[i]);
                }
            }
        }

        virtual ~Layer() {}

        /**
        * @brief Implements common layer setup functionality.
        *
        * @param bottom the preshaped input blobs
        * @param top
        *     the allocated but unshaped output blobs, to be shaped by Reshape
        *
        * Checks that the number of bottom and top blobs is correct.
        * Calls LayerSetUp to do special layer setup for individual layer types,
        * followed by Reshape to set up sizes of top blobs and internal buffers.
        * Sets up the loss weight multiplier blobs for any non-zero loss weights.
        * This method may not be overridden.
        */
        void SetUp(const vector<DataBlock<Dtype>*>& bottom,
          const vector<DataBlock<Dtype>*>& top) {
            LayerSetUp(bottom, top);
            Reshape(bottom, top);
            SetLossWeights(top);
        }



        /**
        * @brief Does layer-specific setup: your layer should implement this function
        *        as well as Reshape.
        *
        * @param bottom
        *     the preshaped input blobs, whose data fields store the input data for
        *     this layer
        * @param top
        *     the allocated but unshaped output blobs
        *
        * This method should do one-time layer specific setup. This includes reading
        * and processing relevent parameters from the <code>layer_param_</code>.
        * Setting up the shapes of top blobs and internal buffers should be done in
        * <code>Reshape</code>, which will be called before the forward pass to
        * adjust the top blob sizes.
        */
        virtual void LayerSetUp(const vector<DataBlock<Dtype>*>& bottom,
          const vector<DataBlock<Dtype>*>& top) {}

        /**
        * @brief Adjust the shapes of top blobs and internal buffers to accommodate
        *        the shapes of the bottom blobs.
        *
        * @param bottom the input blobs, with the requested input shapes
        * @param top the top blobs, which should be reshaped as needed
        *
        * This method should reshape top blobs as needed according to the shapes
        * of the bottom (input) blobs, as well as reshaping any internal buffers
        * and making any other necessary adjustments so that the layer can
        * accommodate the bottom blobs.
        */
        virtual void Reshape(const vector<DataBlock<Dtype>*>& bottom,
          const vector<DataBlock<Dtype>*>& top) = 0;




        /**
           * Called by SetUp to initialize the weights associated with any top blobs in
           * the loss function. Store non-zero loss weights in the diff blob.
        */
        inline void SetLossWeights(const vector<DataBlock<Dtype>*>& top) {
            const int num_loss_weights = layer_param_.loss_weight_size();
            if (num_loss_weights) {
              CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
                  "unspecified or specified once per top blob.";
              for (int top_id = 0; top_id < top.size(); ++top_id) {
                const Dtype loss_weight = layer_param_.loss_weight(top_id);
                if (loss_weight == Dtype(0)) { continue; }

                if (loss_.size() <= top_id) {
                  loss_.resize(top_id + 1, Dtype(0));
                }
                loss_[top_id] = loss_weight;

                const int count = top[top_id]->count();

                Dtype* loss_multiplier = top[top_id]->GetCpuDiff();
                bsy_set(count, loss_weight, loss_multiplier);
              }
            }
        }

        inline Dtype GetLoss(const int top_index) const {
            return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
        }




    protected:
        /** The protobuf that stores the layer parameters */
        LayerParameter layer_param_;

        /** The phase: TRAIN or TEST */
        Phase phase_;

        /** The vector that stores the learnable parameters as a set of blobs. */
        vector<shared_ptr<DataBlock<Dtype>>> data_blocks_;

         /** Vector indicating whether to compute the diff of each param blob. */
        vector<bool> should_propagate_down_;

        /** The vector that indicates whether each top blob has a non-zero weight in
        *  the objective function. */
        vector<Dtype> loss_;

        virtual void ForwardCpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) = 0;

        virtual void ForwardGpu(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) {
                this->ForwardCpu(bottom, top);
        }

        virtual void BackwardCpu(const vector<DataBlock<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<DataBlock<Dtype>*>& bottom) = 0;

        virtual void BackwardGpu(const vector<DataBlock<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<DataBlock<Dtype>*>& bottom) {
                this->BackwardGpu(top, propagate_down, bottom);
        }

        inline Dtype Layer<Dtype>::Forward(const vector<DataBlock<Dtype>*>& bottom,
            const vector<DataBlock<Dtype>*>& top) {
            Dtype loss = 0;
            Reshape(bottom, top);

            if(ProgramCpuMode()) {
                ForwardCpu(bottom, top);
                for (int top_id = 0; top_id < top.size(); ++top_id) {
                    if (!this->GetLoss(top_id)) { continue; }
                    const int count = top[top_id]->count();
                    const Dtype* data = top[top_id]->cpu_data();
                    const Dtype* loss_weights = top[top_id]->cpu_diff();
                    Dtype res = caffe_cpu_dot(count, data, loss_weights);
                    loss += res;

                }
            } else {
                ForwardGpu(bottom, top);
                for (int top_id = 0; top_id < top.size(); ++top_id) {
                    if (!this->GetLoss(top_id)) { continue; }
                    const int count = top[top_id]->count();
                    const Dtype* data = top[top_id]->cpu_data();
                    const Dtype* loss_weights = top[top_id]->cpu_diff();
                    Dtype res = caffe_gpu_dot(count, data, loss_weights);
                    loss += res;

            }
            return loss;
         }

         inline void Layer<Dtype>::BackForward(const vector<DataBlock<Dtype>*>& top,
             const vector<bool>& propagate_down
             const vector<DataBlock<Dtype>*>& bottom) {
             if(ProgramCpuMode()) {
                BackwardCpu(top, propagate_down, bottom);
             } else {
                BackwardGpu(top, propagate_down, bottom);
             }
         }

        FORBID_COPY_AND_ASSIGN(Layer)

}