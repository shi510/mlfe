cmake_minimum_required(VERSION 2.8)

project(mlfe CXX)

set(CMAKE_CXX_STANDARD 11)

include(cmake/Utils.cmake)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
option(BUILD_TEST "Build C++ test binaries (require gtest lib)" OFF)
option(BUILD_APPS "Build mlfe Applications (require opencv)" OFF)

if(MSVC)
    msvc_multi_threaded_static_turn(ON)
endif()

set(LIB_TYPE STATIC)
if(BUILD_SHARED_LIBS)
    set(LIB_TYPE SHARED)
endif(BUILD_SHARED_LIBS)

set(mlfe_include_dirs ${PROJECT_SOURCE_DIR})
set(mlfe_library_dependencies)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/Dependencies.cmake)

add_subdirectory(mlfe)
add_subdirectory(apps)
add_subdirectory(unit_test)
