#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "bsy/util/gpu_relative.hpp"
#include "bsy/proto/bsy.pb.h"
using namespace std;


#define FORBID_COPY_AND_ASSIGN(classname) \
    classname(const classname&) = delete; \
    classname& operator=(const classname&) = delete;


bool ProgramCpuMode() {
    #ifndef CPU_ONLY
    #ifdef HAVE_CUDA
    return false;
    #endif
    #endif
    return true;
}

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<DataBlock<float>*>& bottom, \
      const std::vector<DataBlock<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<DataBlock<double>*>& bottom, \
      const std::vector<DataBlock<double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<DataBlock<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<DataBlock<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<DataBlock<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<DataBlock<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)