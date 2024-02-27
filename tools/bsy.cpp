
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bsy/common.hpp"
#include "bsy/distribute_generator.hpp"

#include "bsy/util/math.hpp"

DEFINE_bool(verbose, true, "Enable verbose output");


using namespace std;

using namespace bsy;

void init_third_party(int *argc, char *** argv)
{
    // init gflags
    gflags::ParseCommandLineFlags(argc, argv, true);

    //init glog
    ::google::InitGoogleLogging(*(argv)[0]);

    // set log level
    google::SetStderrLogging(google::GLOG_INFO);


}
int main(int argc, char** argv)
{
 cout << "hello " << endl;

 init_third_party(&argc, &argv);


  float a = 1.0;
  double b = 2.0;
  bsy::partical_specialization(a);
  bsy::partical_specialization(b);

  DistributeGeneratorParameter param;
  Generator<float >* generator = new GaussianGenerator<float>(param);
  float arr[10] = {0.0f};

  generator->Generate(arr, 10);
  for(auto item :arr ){
    LOG(INFO) << item;
  }
  delete generator;

 return 0;
}