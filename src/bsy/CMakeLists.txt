file(GLOB proto_files proto/*.proto)

message("build proto file")


protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})



# include python files either to force generation
add_library(bsyproto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})

target_link_libraries(bsyproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(bsyproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Bsy_LINKER_LIBS 0 PUBLIC bsyproto) # note, crucial to prepend!

# --[ Bsy library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
pickup_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  bsy_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

message(STATUS "test_srcs: ${test_srcs}")
message(STATUS "srcs: ${srcs}")
message(STATUS "test_cuda: ${test_cuda}")
message(STATUS "cuda: ${cuda}")
message(STATUS "cuda_objs: ${cuda_objs}")
message(STATUS "srcs: ${srcs}")

add_library(bsy ${srcs})

target_link_libraries(bsy ${Bsy_LINKER_LIBS})

target_link_libraries(bsy gflags)

target_link_libraries(bsy glog)

target_include_directories(bsy ${Bsy_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Bsy_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)

target_compile_definitions(bsy ${Bsy_DEFINITIONS})
if(Bsy_COMPILE_OPTIONS)
  target_compile_options(bsy ${Bsy_COMPILE_OPTIONS})
endif()


# ---[ Tests
 add_subdirectory(test)


message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "Bsy_LINKER_LIBS: ${Bsy_LINKER_LIBS}")


install(FILES ${proto_hdrs} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bsy/proto)


