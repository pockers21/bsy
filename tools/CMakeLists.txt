# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(name bsy.bin)



add_executable(${name} bsy.cpp)


message(STATUS "link Bsy_LINK: ${Bsy_LINK} to name: ${name}")
target_link_libraries(${name} ${Bsy_LINK})

set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")

