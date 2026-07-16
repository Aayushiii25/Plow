add_subdirectory(unittests)

add_executable(test_correctness test_correctness.cpp)
target_link_libraries(test_correctness
    PRIVATE
    $<TARGET_OBJECTS:llmengine>
    cublas
    cudart
)