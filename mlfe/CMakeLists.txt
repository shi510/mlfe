set(the_library "mlfe")
set(mlfe_source_files)
set(mlfe_header_files)
set(mlfe_cuda_source_files)

file(GLOB_RECURSE files "*.*")
mlfe_group_files("${files}")

add_subdirectory(core)
add_subdirectory(device_context)
add_subdirectory(math)
add_subdirectory(operators)
add_subdirectory(utils)

include_directories(${mlfe_include_dirs})
if(USE_CUDA)
  cuda_add_library(cuda_objects STATIC ${mlfe_cuda_source_files})
  list(APPEND mlfe_library_dependencies cuda_objects)
endif()
add_library(${the_library} ${LIB_TYPE} ${mlfe_source_files} ${mlfe_header_files})
target_link_libraries(${the_library} ${mlfe_library_dependencies})
mlfe_build_flatbuffers(flat_gen_headers
        "${CMAKE_CURRENT_LIST_DIR}/flatbuffers"
        "${CMAKE_CURRENT_LIST_DIR}/flatbuffers"
        gen_fb_target)
add_dependencies(${the_library} gen_fb_target)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

set_target_properties(${the_library} PROPERTIES FOLDER "mlfe_library")
