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

find_package(flatbuffers QUIET)

if(FLATBUFFERS_FOUND)
  message(STATUS "Found flatbuffers : " ${FLATBUFFERS_INCLUDE_DIRS})
  list(APPEND mlfe_include_dirs ${FLATBUFFERS_INCLUDE_DIRS})
  list(APPEND mlfe_library_dependencies flatbuffers)
else()
  message(STATUS "[Can not find flatbuffers. Using third party dir.]")
  set(FLATBUFFERS_CODE_COVERAGE OFF)
  set(FLATBUFFERS_BUILD_TESTS OFF)
  set(FLATBUFFERS_INSTALL OFF)
  set(FLATBUFFERS_BUILD_FLATLIB ON)
  set(FLATBUFFERS_BUILD_FLATC ON)
  set(FLATBUFFERS_BUILD_FLATHASH OFF)
  set(FLATBUFFERS_BUILD_GRPCTEST OFF)
  set(FLATBUFFERS_BUILD_SHAREDLIB OFF)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/flatbuffers)
  list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include)
  list(APPEND mlfe_library_dependencies flatbuffers)
endif()

if(BUILD_APPS)
  find_package(OpenCV REQUIRED)
endif()

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
    	list(APPEND mlfe_library_dependencies ${CUDA_CUBLAS_LIBRARIES})
    	list(APPEND mlfe_include_dirs ${CUDA_INCLUDE_DIRS})
      list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/cub)
    endif()
endif()
