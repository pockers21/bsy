################################################################################################
# Defines global Bsy_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(set_link)
  if(BUILD_SHARED_LIBS)
    set(Bsy_LINK bsy)
  else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(Bsy_LINK -Wl,-force_load bsy)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(Bsy_LINK -Wl,--whole-archive bsy -Wl,--no-whole-archive)
    endif()
  endif()

endmacro()


################################################################################################
# Short command getting bsy sources (assuming standard Bsy code tree)
# Usage:
#   bsy_pickup_sources(<root>)
function(pickup_sources root)

  # collect files
  file(GLOB test_hdrs    ${root}/include/bsy/test/test_*.h*)
  file(GLOB test_srcs    ${root}/src/bsy/test/test_*.cpp)
  file(GLOB_RECURSE hdrs ${root}/include/bsy/*.h*)
  file(GLOB_RECURSE srcs ${root}/src/bsy/*.cpp)


  message(STATUS "hdrs: ${hdrs}")
  message(STATUS "test_hdrs: ${test_hdrs}")

  list(REMOVE_ITEM  hdrs ${test_hdrs})
  list(REMOVE_ITEM  srcs ${test_srcs})

  # adding headers to make the visible in some IDEs (Qt, VS, Xcode)
  list(APPEND srcs ${hdrs} ${PROJECT_BINARY_DIR}/bsy_config.h)
  list(APPEND test_srcs ${test_hdrs})


  # collect cuda files
  file(GLOB    test_cuda ${root}/src/bsy/test/test_*.cu)
  file(GLOB_RECURSE cuda ${root}/src/bsy/*.cu)

  message(STATUS "cuda: ${cuda}")
  message(STATUS "test_cuda: ${test_cuda}")
  list(REMOVE_ITEM  cuda ${test_cuda})

  # add proto to make them editable in IDEs too
  file(GLOB_RECURSE proto_files ${root}/src/bsy/*.proto)
  list(APPEND srcs ${proto_files})

  # convert to absolute paths
  convert_absolute_paths(srcs)
  convert_absolute_paths(cuda)
  convert_absolute_paths(test_srcs)
  convert_absolute_paths(test_cuda)

  # propagate to parent scope
  set(srcs ${srcs} PARENT_SCOPE)
  set(cuda ${cuda} PARENT_SCOPE)
  set(test_srcs ${test_srcs} PARENT_SCOPE)
  set(test_cuda ${test_cuda} PARENT_SCOPE)
endfunction()
