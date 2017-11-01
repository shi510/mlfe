cmake_minimum_required(VERSION 2.8)

project(mlfe CXX)

set(CMAKE_CXX_STANDARD 11)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)

set(LIB_TYPE STATIC)
if(BUILD_SHARED_LIBS)
    set(LIB_TYPE SHARED)
endif(BUILD_SHARED_LIBS)

set(mlfe_include_dirs ${PROJECT_SOURCE_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/Dependencies.cmake)
include(cmake/Utils.cmake)

add_subdirectory(mlfe)
