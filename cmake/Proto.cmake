# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

find_package( Protobuf REQUIRED )
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${PROTOBUF_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${PROTOBUF_LIBRARIES})


# place where to generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/bsy/proto")
include_directories("${PROJECT_BINARY_DIR}/include")



################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   protobuf_generate_cpp_py(<output_dir> <srcs_var> <hdrs_var> <python_var> <proto_files>)
function(protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  message(STATUS "PROTOBUF_IMPORT_DIRS: ${PROTOBUF_IMPORT_DIRS}")
  if(DEFINED PROTOBUF_IMPORT_DIRS)

    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})

  add_custom_command(
    OUTPUT output.txt
    COMMAND echo "Hello, World!" > output.txt
    COMMENT "Generating output.txt"
  )

  message(STATUS "output_dir: ${output_dir}")
  message(STATUS "abs_fil: ${abs_fil}")
  message(STATUS "_protoc_include: ${_protoc_include}")


  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
             "${output_dir}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${PROJECT_BINARY_DIR}/include --proto_path ${PROJECT_SOURCE_DIR}/src ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
