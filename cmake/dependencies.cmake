find_package(Threads)
if(THREADS_FOUND)
  if(${CMAKE_USE_WIN32_THREADS_INIT})
    message(STATUS "Found Threads : " "win32 thread")
  elseif(${CMAKE_USE_WIN32_THREADS_INIT})
    message(STATUS "Found Threads : " "pthread")
  endif()
  list(APPEND mlfe_library_dependencies ${CMAKE_THREAD_LIBS_INIT})
endif()

find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  message(STATUS "Found Eigen3 : " ${EIGEN3_INCLUDE_DIRS})
  list(APPEND mlfe_include_dirs ${EIGEN3_INCLUDE_DIRS})
else()
  message(STATUS "[Can not find Eigen3. Using third party dir.]")
  list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/eigen)
endif()

if(BUILD_TEST)
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_GTEST ON)
  set(INSTALL_GTEST OFF)
  set(BUILD_GMOCK OFF)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# find_package(flatbuffers QUIET)
# if(FLATBUFFERS_FOUND)
#   message(STATUS "Found flatbuffers : " ${FLATBUFFERS_INCLUDE_DIRS})
#   list(APPEND mlfe_include_dirs ${FLATBUFFERS_INCLUDE_DIRS})
#   list(APPEND mlfe_library_dependencies flatbuffers)
# else()
#   message(STATUS "[Can not find flatbuffers. Using third party dir.]")
#   set(FLATBUFFERS_CODE_COVERAGE OFF CACHE BOOL "FLATBUFFERS_CODE_COVERAGE" FORCE)
#   set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "FLATBUFFERS_BUILD_TESTS" FORCE)
#   set(FLATBUFFERS_INSTALL OFF CACHE BOOL "FLATBUFFERS_INSTALL" FORCE)
#   set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "FLATBUFFERS_BUILD_FLATLIB" FORCE)
#   set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "FLATBUFFERS_BUILD_FLATC" FORCE)
#   set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATHASH" FORCE)
#   set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "FLATBUFFERS_BUILD_GRPCTEST" FORCE)
#   set(FLATBUFFERS_BUILD_SHAREDLIB OFF CACHE BOOL "FLATBUFFERS_BUILD_SHAREDLIB" FORCE)
#   add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/flatbuffers)
#   list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include)
#   list(APPEND mlfe_library_dependencies flatbuffers)
# endif()

find_package(Protobuf "3.11.4")

if(${PROTOBUF_FOUND} AND ${USE_INSTALLED_LIBRARY})
  message(STATUS "Found Protobuf on your system : " ${Protobuf_INCLUDE_DIRS})
  list(APPEND mlfe_include_dirs ${Protobuf_INCLUDE_DIRS})
  list(APPEND mlfe_library_dependencies ${Protobuf_LIBRARIES})
else()
  message(STATUS "[Protobuf will be compiled which is in third party folder.]")
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "protobuf_BUILD_TESTS" FORCE)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/protobuf/cmake)
  list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/protobuf/src)
  list(APPEND mlfe_library_dependencies libprotobuf)
  set(Protobuf_PROTOC_EXECUTABLE protoc)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/cudnn.cmake)

if(USE_CUDA OR USE_CUDNN)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        FindCudnn(${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
        list(APPEND mlfe_library_dependencies ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDNN_LIBRARIES})
        list(APPEND mlfe_include_dirs ${CUDA_INCLUDE_DIRS})
        list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/cub)
        # set(ARCH_LIST 35 37 50 52 53 60 61 62 70)
        # foreach(var ${ARCH_LIST})
        #   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${var},code=sm_${var}")
        # endforeach(var)
    endif()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/mkldnn.cmake)
if(USE_INTEL_MKLDNN)
    find_mkldnn()
    if(${MKLDNN_FOUND})
        message("Found MKLDNN include dirs: " ${MKLDNN_INCLUDE_DIRS})
        message("Found MKLDNN libs: " ${MKLDNN_LIBS})
        list(APPEND mlfe_include_dirs ${MKLDNN_INCLUDE_DIRS})
        list(APPEND mlfe_library_dependencies ${MKLDNN_LIBS})
    else()
        set(INTEL_MKLDNN_ROOT ${PROJECT_SOURCE_DIR}/third_party/mkldnn)
        message(STATUS "Not found intel-mkldnn, use instead third_party/mkldnn")
        message(STATUS "Downloading intel mkl...")
        if(WIN32)
            execute_process(
                COMMAND ${INTEL_MKLDNN_ROOT}/scripts/prepare_mkl.bat
                )
        else()
            execute_process(
                COMMAND bash ${INTEL_MKLDNN_ROOT}/scripts/prepare_mkl.sh
                )
        endif()
        set(WITH_EXAMPLE OFF CACHE BOOL "INTEL MKLDNN EXAMPLE BUILD" FORCE)
        set(WITH_TEST OFF CACHE BOOL "INTEL MKLDNN TEST BUILD" FORCE)
        add_subdirectory(${INTEL_MKLDNN_ROOT})
        list(APPEND mlfe_include_dirs ${INTEL_MKLDNN_ROOT}/include)
        list(APPEND mlfe_library_dependencies mkldnn ${MKLLIB})
    endif()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/xnnpack.cmake)
if(USE_XNNPACK AND ${USE_INSTALLED_LIBRARY})
  find_xnnpack()
  if(${XNNPACK_FOUND})
      message("Found XNNPACK include dirs: " ${XNNPACK_INCLUDE_DIRS})
      message("Found XNNPACK libs: " ${XNNPACK_LIBS})
      list(APPEND mlfe_include_dirs ${XNNPACK_INCLUDE_DIRS})
      list(APPEND mlfe_library_dependencies ${XNNPACK_LIBS})
  endif()
elseif(USE_XNNPACK)
  set(XNNPACK_ROOT ${PROJECT_SOURCE_DIR}/third_party/XNNPACK)
  set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "XNNPACK BENCHMARKS BUILD" FORCE)
  set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "XNNPACK TESTS BUILD" FORCE)
  add_subdirectory(${XNNPACK_ROOT})
  list(APPEND mlfe_include_dirs ${XNNPACK_ROOT}/include)
  list(APPEND mlfe_library_dependencies XNNPACK)
endif()

if(UNIX AND NOT APPLE AND NOT MLFE_LITE)
    list(APPEND mlfe_library_dependencies stdc++fs)
endif()
