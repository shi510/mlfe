
function(mlfe_group_files files)
    foreach(FILE ${files})
        # Get the directory of the source file
        get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)

        # Remove common directory prefix to make the group
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" GROUP "${PARENT_DIR}")

        # Make sure we are using windows slashes
        string(REPLACE "/" "\\" GROUP "${GROUP}")

        # Group into "Source Files" and "Header Files"
        if ("${FILE}" MATCHES ".*\\.[cpp|h]")
            set(GROUP "${GROUP}")
        endif()

        source_group("${GROUP}" FILES "${FILE}")
    endforeach()
endfunction()

function(msvc_multi_threaded_static_turn on_or_off)
    if(on_or_off)
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    else()
        string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} PARENT_SCOPE)
endfunction()
