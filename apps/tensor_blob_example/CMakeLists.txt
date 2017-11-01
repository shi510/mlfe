set(the_app tensor_example)

include_directories(${mlfe_include_dirs})

add_executable(${the_app} tensor_example.cpp)

target_link_libraries(${the_app} mlfe)

set_target_properties(${the_app} PROPERTIES FOLDER "apps")
