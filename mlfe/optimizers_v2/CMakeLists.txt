file(GLOB srcs "*.cc")
file(GLOB hdrs "*.h")

if(USE_CUDA OR USE_CUDNN)
    file(GLOB target_src "impl/cuda/*.cc")
else()
    file(GLOB target_src "impl/cpu/*.cc")
endif()

list(APPEND mlfe_source_files ${srcs} ${target_src})
list(APPEND mlfe_header_files ${hdrs})
set(mlfe_source_files ${mlfe_source_files} PARENT_SCOPE)
set(mlfe_header_files ${mlfe_header_files} PARENT_SCOPE)
