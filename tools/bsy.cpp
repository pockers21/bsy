
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bsy/common.hpp"

#include "bsy/util/math.hpp"

DEFINE_bool(verbose, true, "Enable verbose output");


using namespace std;

void init_third_party(int *argc, char *** argv)
{
    // init gflags
    gflags::ParseCommandLineFlags(argc, argv, true);

    //init glog
    ::google::InitGoogleLogging(*(argv)[0]);

}
int main(int argc, char** argv)
{
 cout << "hello " << endl;

 init_third_party(&argc, &argv);



  // 使用命令行参数
  if (FLAGS_verbose) {
    std::cout << "Verbose mode is enabled." << std::endl;
  } else {
    std::cout << "Verbose mode is disabled." << std::endl;
  }


  float a = 1.0;
  double b = 2.0;
  partical_specialization(a);
  partical_specialization(b);


 return 0;
}