cmake_minimum_required(VERSION 2.8.7)


if(POLICY CMP0053)
  cmake_policy(SET CMP0053 NEW)
endif()

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW)
endif()


project(bsy C CXX)


set(BSY_TARGET_VERSION "1.0.0" CACHE STRING "Bsy logic version")

add_definitions(-DBSY_VERSION=${BSY_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/find)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")


#only support unix platform
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FARAL_ERROR "This project requires Linux to build.")
    return()
endif()

include(ExternalProject)
include(GNUInstallDirs)

message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES: ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
message("CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")

include(cmake/Util.cmake)
include(cmake/Preprocess.cmake)
include(cmake/Proto.cmake)
include(cmake/Targets.cmake)



option(CPU_ONLY "Build bsy without cuda" OFF)
option(USE_CUDNN "Build bsy with cudnn" OFF)
option(USE_NCCL "Build bsy with nccl" OFF)

option(BUILD_SHARED_LIBS "Build bsy with nccl" ON)
option(BUILD_python "Build bsy with nccl" ON)
set(python_version "3" CACHE STRING "Specify which Python version to use")


option(USE_OPENCV "Build bsy with opencv " ON)
option(USE_LEVELDB "Build bsy with level db" ON)
#option(USE_LMDB "Build bsy with lmdb" ON)
#option(USE_OPENMP "Build bsy with openmp" ON)
#option(USE_HDF5 "Build bsy with hdf5" ON)


include(cmake/Dependencies.cmake)

message( "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message( "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message( "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT DEFINED")
 else()
   message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT not DEFINED")
 endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT ok")
 else()
   message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT not ok")
 endif()

 if(CMAKE_BUILD_TYPE)
   message("CMAKE_BUILD_TYPE ok")
 else()
   message("CMAKE_BUILD_TYPE not ok")
 endif()

 if(DEFINED CMAKE_BUILD_TYPE)
   message("CMAKE_BUILD_TYPE DEFINED")
 else()
   message("CMAKE_BUILD_TYPE not DEFINED")
 endif()


# ---[ Flags
if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

set_link()


# ---[ Includes
set(Bsy_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Bsy_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})


message("PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")

# ---[ Config generation
configure_file(cmake/template/bsy_config.h.in "${PROJECT_BINARY_DIR}/bsy_config.h")



if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Bsy_ALL_INCLUDE_DIRS ${Bsy_INCLUDE_DIRS})
  list(REMOVE_ITEM Bsy_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Bsy_INCLUDE_DIR} ${Bsy_SRC_DIR} ${Bsy_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Bsy_ALL_DEFINITIONS ${Bsy_DEFINITIONS})
  message("Bsy_ALL_DEFINITIONS: ${Bsy_ALL_DEFINITIONS}")
  list(REMOVE_ITEM Bsy_ALL_DEFINITIONS PRIVATE PUBLIC)
  message("After removing, Caffe_ALL_DEFINITIONS: ${Bsy_ALL_DEFINITIONS}")
  list(APPEND CUDA_NVCC_FLAGS ${Bsy_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/bsy)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)

#message(SEND_ERROR "srcs: ${srcs}")

