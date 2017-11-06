set(the_app test_softmax_xent)

include_directories(${mlfe_include_dirs})

add_executable(${the_app} test_softmax_xent.cpp)

target_link_libraries(${the_app} mlfe gtest_main)

set_target_properties(${the_app} PROPERTIES FOLDER "test_unit")
