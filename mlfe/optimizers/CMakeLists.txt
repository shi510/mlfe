file(GLOB srcs "*.cc")
file(GLOB_RECURSE hdrs "*.h")

if(USE_CUDA OR USE_CUDNN)
    file(GLOB target_src "impl/cuda/*.cc")
    file(GLOB cuda_srcs "impl/cuda/kernel/*.cu")
    list(APPEND mlfe_cuda_source_files ${cuda_srcs})
    set(mlfe_cuda_source_files ${mlfe_cuda_source_files} PARENT_SCOPE)
else()
    file(GLOB target_src "impl/cpu/*.cc")
endif()

list(APPEND mlfe_source_files ${srcs} ${target_src})
list(APPEND mlfe_header_files ${hdrs})
set(mlfe_source_files ${mlfe_source_files} PARENT_SCOPE)
set(mlfe_header_files ${mlfe_header_files} PARENT_SCOPE)
