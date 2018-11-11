cmake_minimum_required(VERSION 2.8)

project(mlfe CXX)

set(CMAKE_CXX_STANDARD 11)

include(cmake/Utils.cmake)

if(UNIX)
  option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
endif()
option(BUILD_TEST "Build C++ test binaries (require gtest lib)" OFF)
option(BUILD_EXAMPLE "Build mlfe EXAMPLE (require opencv)" OFF)
option(USE_CUDA "NVIDIA CUDA USE" OFF)

if(MSVC)
    msvc_multi_threaded_static_turn(ON)
endif()

set(LIB_TYPE STATIC)
if(BUILD_SHARED_LIBS)
    set(LIB_TYPE SHARED)
endif()

set(mlfe_include_dirs ${PROJECT_SOURCE_DIR})
set(mlfe_library_dependencies)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/Dependencies.cmake)

add_subdirectory(mlfe)
add_subdirectory(example)
add_subdirectory(unit_test)
