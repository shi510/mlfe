cmake_minimum_required(VERSION 3.13)

project(mlfe VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(cmake/utils.cmake)

if(UNIX)
  option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
endif()
option(BUILD_TEST "Build C++ test binaries (require gtest lib)" OFF)
option(BUILD_EXAMPLE "Build mlfe EXAMPLE (require opencv)" OFF)
option(MLFE_LITE "Only inference features are included." OFF)
option(USE_CUDA "NVIDIA CUDA USE" OFF)
option(USE_CUDNN "NVIDIA CUDNN USE" OFF)
option(USE_INTEL_MKLDNN "INTEL MKL-DNN Library USE" OFF)
option(USE_XNNPACK "XNNPACK Library USE" OFF)
option(USE_INSTALLED_LIBRARY "Build with installed library on your system" OFF)

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

include(cmake/dependencies.cmake)

add_subdirectory(mlfe)
add_subdirectory(example)
add_subdirectory(unit_test)
