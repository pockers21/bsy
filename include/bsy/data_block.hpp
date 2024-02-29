#pragma once
#include "bsy/common.hpp"
#include "bsy/memory_block.hpp"

namespace bsy{

template <typename Dtype>

class DataBlock{
    public:
        DataBlock();
        DataBlock(const int num, const int channal, const int height, const int width);
        explicit DataBlock(const vector<int> & shape);

        void  Reshape(const vector<int> shape);
        void  Reshape(const int num, const int channal, const int height, const int width);
        void  Reshape(const DataBlock& other);
        inline string Shape_String(){
            ostringstream stream;
            for(int index = 0 ; index < shape_.size(); index++){
                stream << shape_[index] << " ";

            }
            stream << "(" << count_ << ")";
            return stream.str();
        }

        inline const vector<int>& GetShape() const {return shape_;}

        inline int GetAxes() const { return shape_.size();}

        inline int GetCount() const { return count_;}

        Dtype * GetCpuData() const ;
        void SetCpuData(Dtype* data);

        Dtype * GetCpuDiff() const ;
        void SetCpuDiff(Dtype* data);

        Dtype * GetGpuData() const ;
        void SetGpuData(Dtype* data);

        Dtype * GetGpuDiff() const ;
        void SetGpuDiff(Dtype* data);

        inline int shape(const int index) const{
            CHECK_GE(index, -GetAxes())
                << "axis " << index << " out of range for " << GetAxes()
                << "-D Blob with shape " << Shape_String();

            CHECK_LT(index, GetAxes())
                << "axis " << index << " out of range for " << GetAxes()
                << "-D Blob with shape " << Shape_String();

            if(index < 0){
                index += GetAxes();
            }

            return index;
        }

        inline int num() const {return shape(0); }

        inline int channals() const {return shape(1); }

        inline int height() const {return shape(2); }

        inline int width() const {return shape(3); }


        inline int offset(const int num, const int channal, const int height, const int width) const {
            CHECK_GE(num, 0);
            CHECK_LE(num, this->num());
            CHECK_GE(this->channels(), 0);
            CHECK_LE(channal, this->channels());
            CHECK_GE(this->height(), 0);
            CHECK_LE(height, this->height());
            CHECK_GE(this->width(), 0);
            CHECK_LE(width, this->width());
            return ((num * this->channels() + channal) * this->height() + height) * this->width() + width;
        }

        inline int offset(const vector<int> indices) const {
            CHECK_LE(indices.size(), GetAxes());
            int offset = 0;
            for (int i = 0; i < GetAxes(); ++i) {
              offset *= shape(i);
              if (indices.size() > i) {
                CHECK_GE(indices[i], 0);
                CHECK_LT(indices[i], shape(i));
                offset += indices[i];
              }
            }
            return offset;
        }

        Dtype DataAt(const int num, const int channal, const int height, const int width) const{
            return GetCpuData()[offset(num, channal, height, width)];
        }
        Dtype DataAt(const vector<int> index) const {
            return GetCpuData()[offset(index)];
        }

        Dtype DiffAt(const int num, const int channal, const int height, const int width) const {
            return GetCpuDiff()[offset(num, channal, height, width)];
        }
        Dtype DidffAt(const vector<int> index) const {
            return GetCpuDiff()[offset(index)];
        }

        const shared_ptr<MemoryBlock<Dtype>>& GetDataMemBlock() const{
            return data_;
        }

        const shared_ptr<MemoryBlock<Dtype>>& GetDiffMemBlock() const{
            return diff_;
        }

        bool ShapeEqualsWithProto(const DataBlockProto& other){
            vector<int> other_shape(other.shape().dim_size());
            for (int i = 0; i < other.shape().dim_size(); ++i) {
              other_shape[i] = other.shape().dim(i);
            }
            return shape_ == other_shape;
        }

        void ShareData(const DataBlock& other);

        void ShareDiff(const DataBlock& other);

        inline int count(int start_axis, int end_axis) const {
            CHECK_LE(start_axis, end_axis);
            CHECK_GE(start_axis, 0);
            CHECK_GE(end_axis, 0);
            CHECK_LE(start_axis, GetAxes());
            CHECK_LE(end_axis, GetAxes());
            int count = 1;

            for (int i = start_axis; i < end_axis; ++i) {
              count *= shape(i);
            }
            return count;
        }

        inline int count(int start_axis) const {
            return count(start_axis, GetAxes());
        }

        void FromProto(const DataBlockProto& proto, bool reshape) {
            if(reshape) {
                vector<int> shape;
                shape.resize(proto.shape().dim_size());
                for (int i = 0; i < proto.shape().dim_size(); ++i) {
                    shape[i] = proto.shape().dim(i);
                }
                this->Reshape(shape);
            } else {
               CHECK(this->ShapeEqualsWithProto(proto))
            }
            Dtype* data_vec = GetCpuData();
            if (proto.double_data_size() > 0) {
                CHECK_EQ(count_, proto.double_data_size());
                for (int i = 0; i < count_; ++i) {
                  data_vec[i] = proto.double_data(i);
                }
            } else {
                CHECK_EQ(count_, proto.data_size());
                for (int i = 0; i < count_; ++i) {
                  data_vec[i] = proto.data(i);
                }
            }

            if (proto.double_diff_size() > 0) {
                CHECK_EQ(count_, proto.double_diff_size());
                Dtype* diff_vec = mutable_cpu_diff();
                for (int i = 0; i < count_; ++i) {
                  diff_vec[i] = proto.double_diff(i);
                }
            } else if (proto.diff_size() > 0) {
                CHECK_EQ(count_, proto.diff_size());
                Dtype* diff_vec = mutable_cpu_diff();
                for (int i = 0; i < count_; ++i) {
                  diff_vec[i] = proto.diff(i);
                }
            }
        }


        void ToProto(BlobProto* proto, bool write_diff = false) const;
    private:

        int num_;
        int channal_;
        int height_;
        int width_;

        vector <int> shape_;
        int count_;
        int capacity_;
        shared_ptr<MemoryBlock<Dtype>> data_;
        shared_ptr<MemoryBlock<Dtype>> diff_;

        FORBID_COPY_AND_ASSIGN(DataBlock)

};


}