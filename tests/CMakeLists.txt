add_executable(tinyinfer_tests scaffold_test.cpp)
target_link_libraries(tinyinfer_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tests COMMAND tinyinfer_tests)
