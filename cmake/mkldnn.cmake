function(find_mkldnn )
    set(MKLDNN_FOUND FALSE)
    find_path(MKLDNN_INCLUDE_DIR mkldnn.h)
    find_library(MKLDNN_LIB mkldnn)

    if(NOT MKLDNN_INCLUDE_DIR-NOTFOUND AND NOT MKLDNN_LIBS-NOTFOUND)
        set(MKLDNN_FOUND TRUE)
    endif()

    if(MKLDNN_FOUND)
        set(MKLDNN_INCLUDE_DIRS ${MKLDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(MKLDNN_LIBS ${MKLDNN_LIBS} PARENT_SCOPE)
    endif()
endfunction()
