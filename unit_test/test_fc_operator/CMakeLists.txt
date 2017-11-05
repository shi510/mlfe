set(the_app test_fc_operator)
message("test_fc_operator : " ${mlfe_include_dirs})
include_directories(${mlfe_include_dirs})

add_executable(${the_app} test_fc_operator.cpp)

target_link_libraries(${the_app} mlfe gtest_main)

set_target_properties(${the_app} PROPERTIES FOLDER "test_unit")
