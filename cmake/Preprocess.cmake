# ---[ Configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)


if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# --[ assume Release default
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "CMAKE_BUILD_TYPE equals with empty str")
  set(CMAKE_BUILD_TYPE Release)
endif()


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE)
endif()

#set CMAKE_INSTALL_RPATH
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
     ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} __find_install_dir_in_link_dir)
if(${__find_install_dir_in_link_dir} STREQUAL -1)
  message(STATUS "find ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} in CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES failed")
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()

#set debug postfix
set(Bsy_DEBUG_POSTFIX "-d")


