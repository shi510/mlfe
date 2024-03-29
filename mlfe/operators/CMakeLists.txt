file(GLOB srcs "*.cc")
file(GLOB hdrs "*.h")

if(USE_XNNPACK)
    file(GLOB target_src "impl/xnnpack/*.cc")
elseif(USE_INTEL_MKLDNN)
    file(GLOB target_src "impl/mkldnn/*.cc")
elseif((USE_CUDA AND USE_CUDNN) OR USE_CUDNN)
    file(GLOB target_src "impl/cudnn/*.cc")
    file(GLOB cuda_srcs "impl/cuda/kernel/*.cu")
    list(APPEND mlfe_cuda_source_files ${cuda_srcs})
    set(mlfe_cuda_source_files ${mlfe_cuda_source_files} PARENT_SCOPE)
elseif(USE_CUDA AND NOT USE_CUDNN)
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
