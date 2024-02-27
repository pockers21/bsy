# These lists are later turned into target properties on main bsy library target
set(Bsy_LINKER_LIBS "")
set(Bsy_INCLUDE_DIRS "")
set(Bsy_DEFINITIONS "")
set(Bsy_COMPILE_OPTIONS "")

# ---[ Boost
find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem)
list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
list(APPEND Bsy_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

# ---[ OpenMP
if(USE_OPENMP)
  # Ideally, this should be provided by the BLAS library IMPORTED target. However,
  # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
  # to flick the switch manually as needed.
  #
  # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
  # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
  # However, this naïve method will force any user of Bsy to add the same kludge
  # into their buildsystem again, so we put these options into per-target PUBLIC
  # compile options and link flags, so that they will be exported properly.
  find_package(OpenMP REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Bsy_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

find_package(gflags REQUIRED)

find_package(glog REQUIRED)

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
list(APPEND Bsy_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})


if(USE_HDF5)
  find_package(HDF5 COMPONENTS HL REQUIRED)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  add_definitions(-DUSE_HDF5)
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
  list(APPEND Bsy_DEFINITIONS PUBLIC -DUSE_LMDB)
endif()
message(STATUS "LMDB_DIR end: ${LMDB_DIR}")

message(STATUS "LEVELDB_DIR: ${LMDB_DIR}")

# ---[ LEVELDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
  list(APPEND Bsy_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif()


# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${Snappy_LIBRARIES})
endif()

include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
  endif()

  list(APPEND Bsy_DEFINITIONS PUBLIC -DCPU_ONLY)
endif()

if(USE_NCCL)
  find_package(NCCL REQUIRED)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Bsy_LINKER_LIBS ${NCCL_LIBRARIES})
  add_definitions(-DUSE_NCCL)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND Bsy_DEFINITIONS PUBLIC -DUSE_OPENCV)
endif()

# ---[ BLAS
set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
  find_package(Atlas REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
  find_package(OpenBLAS REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
  find_package(MKL REQUIRED)
  list(APPEND Bsy_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
  list(APPEND Bsy_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
  list(APPEND Bsy_DEFINITIONS PUBLIC -DUSE_MKL)
endif()

# ---[ Python
if(BUILD_python)
  message(STATUS "python_version: ${python_version}")
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)

    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

      STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    message(FATAL_ERROR "bsy does not support python version less than 3.0.0")
  endif()

  message(STATUS "Boost_PYTHON_FOUND: ${Boost_PYTHON_FOUND}")
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(BUILD_python_layer)
      list(APPEND Bsy_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
      list(APPEND Bsy_INCLUDE_DIRS PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} PUBLIC ${Boost_INCLUDE_DIRS})
      list(APPEND Bsy_LINKER_LIBS PRIVATE ${PYTHON_LIBRARIES} PUBLIC ${Boost_LIBRARIES})
    endif()
  endif()
endif()


message(STATUS "Bsy_DEFINITIONS: ${Bsy_DEFINITIONS}")

message(STATUS "Bsy_LINKER_LIBS: ${Bsy_LINKER_LIBS}")
message(STATUS "Bsy_INCLUDE_DIRS: ${Bsy_INCLUDE_DIRS}")
message(STATUS "Bsy_COMPILE_OPTIONS: ${Bsy_COMPILE_OPTIONS}")