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