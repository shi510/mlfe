find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  message(STATUS "Found Eigen3 : " ${EIGEN3_INCLUDE_DIRS})
  list(APPEND mlfe_include_dirs ${EIGEN3_INCLUDE_DIRS})
else()
  message(STATUS "[Can not find Eigen3. Using third party dir.]")
  list(APPEND mlfe_include_dirs ${PROJECT_SOURCE_DIR}/third_party/eigen)
endif()
