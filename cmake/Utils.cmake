
function(mlfe_group_files files)
    foreach(FILE ${files})
        # Get the directory of the source file
        get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)

        # Remove common directory prefix to make the group
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" GROUP "${PARENT_DIR}")

        # Make sure we are using windows slashes
        string(REPLACE "/" "\\" GROUP "${GROUP}")

        # Group into "Source Files" and "Header Files"
        if ("${FILE}" MATCHES ".*\\.[cpp|h|cu]")
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

function(mlfe_build_flatbuffers generated_headers schema_include_dirs generated_includes_dir custom_target_name)
    set(${generated_headers})
    foreach(include_dir ${schema_include_dirs})
        file(GLOB_RECURSE schemas "${include_dir}/*.fbs")
        string(REPLACE ".fbs" "_generated.h" generated_header "${schemas}")
        add_custom_command(OUTPUT ${generated_header}
          COMMAND flatc -c -o ${generated_includes_dir} ${schemas}
          COMMENT "Building C++ header for ${schemas}"
        )
        list(APPEND ${generated_headers} ${generated_header})
    endforeach()
    add_custom_target(${custom_target_name} DEPENDS ${${generated_headers}})
    set(${generated_headers} ${${generated_headers}}  PARENT_SCOPE)
endfunction()

function(generate_proto_cpp dep gen_srcs gen_hdrs proto_path)
    set(protoc_args --proto_path=${proto_path} --cpp_out=${proto_path})
    file(GLOB proto_files ${proto_path}/*.proto*)
    # Remove not protobuf files.
    # For example, a.proto3.cc, which is created after building this project.
    # It should be contained only files with .proto or .proto[0-9] extension.
    list(FILTER proto_files INCLUDE REGEX ".+\.proto[0-9]?$")
    set(srcs ${proto_files})
    set(hdrs ${proto_files})
    list(TRANSFORM srcs REPLACE "\\.proto[0-9]?" "\.pb.cc")
    list(TRANSFORM hdrs REPLACE "\\.proto[0-9]?" "\.pb.h")
    add_custom_command(OUTPUT ${srcs} ${hdrs}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} ${protoc_args} ${proto_files}
        COMMENT "proto generate cpp for ${proto_files}"
    )
    add_custom_target(${dep} DEPENDS ${srcs} ${hdrs})
    set(${gen_srcs} ${srcs} PARENT_SCOPE)
    set(${gen_hdrs} ${hdrs} PARENT_SCOPE)
endfunction()