function(FindCudnn CUDA_INCLUDE_PATH, CUDA_LIB_PATH)
    set(CUDNN_FOUND FALSE)
    find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDA_INCLUDE_PATH})
    if(MSVC)
        find_library(CUDNN_LIBRARY_RELEASE cudnn.lib
            PATHS ${CUDA_LIB_PATH} PATH_SUFFIXES Release)

        find_library(CUDNN_LIBRARY_DEBUG cudnn.lib
            PATHS ${CUDA_LIB_PATH} PATH_SUFFIXES Debug)

        set(CUDNN_LIBRARY optimized ${CUDNN_LIBRARY_RELEASE} debug ${CUDNN_LIBRARY_DEBUG})
    else()
        find_library(CUDNN_LIBRARY cudnn PATHS ${CUDA_LIB_PATH})
    endif()

    if(NOT CUDNN_INCLUDE_DIR-NOTFOUND AND NOT CUDNN_LIBRARY_RELEASE-NOTFOUND)
        set(CUDNN_FOUND TRUE)
    endif()

    if(CUDNN_FOUND)
        set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARY} PARENT_SCOPE)
    endif()
endfunction()
