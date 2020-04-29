set(the_library "mlfe")
set(mlfe_source_files)
set(mlfe_header_files)
set(mlfe_cuda_source_files)

file(GLOB_RECURSE files "*.*")
mlfe_group_files("${files}")

add_subdirectory(core)
add_subdirectory(device_context)
add_subdirectory(math)
add_subdirectory(module)
add_subdirectory(operators)
add_subdirectory(optimizers)
add_subdirectory(utils)

list(APPEND mlfe_header_files core.h)
list(APPEND mlfe_header_files operators.h)
list(APPEND mlfe_header_files optimizers.h)

include_directories(${mlfe_include_dirs})
if(USE_CUDA)
  cuda_add_library(cuda_objects STATIC ${mlfe_cuda_source_files})
  list(APPEND mlfe_library_dependencies cuda_objects)
endif()
add_library(${the_library} ${LIB_TYPE} ${mlfe_source_files} ${mlfe_header_files})
target_link_libraries(${the_library} ${mlfe_library_dependencies})
if(USE_CUDNN)
    target_compile_definitions(${the_library} PRIVATE OPTION_USE_CUDNN)
elseif(USE_CUDA)
    target_compile_definitions(${the_library} PRIVATE OPTION_USE_CUDA)
elseif(USE_INTEL_MKLDNN)
    target_compile_definitions(${the_library} PRIVATE OPTION_USE_MKLDNN)
endif()
mlfe_build_flatbuffers(flat_gen_headers
        "${CMAKE_CURRENT_LIST_DIR}/flatbuffers"
        "${CMAKE_CURRENT_LIST_DIR}/flatbuffers"
        gen_fb_target)
add_dependencies(${the_library} gen_fb_target)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING REGEX ".+\\.[hpp|h]")

install(TARGETS ${the_library}
        DESTINATION lib)

set_target_properties(${the_library} PROPERTIES FOLDER "mlfe_library")
set_target_properties(${the_library} PROPERTIES DEBUG_POSTFIX "d")
