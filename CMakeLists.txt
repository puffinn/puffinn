cmake_minimum_required(VERSION 3.5)
project(Puffinn CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -O3 -g -fopenmp")

include_directories("include")

add_executable(GloveExample "examples/glove.cpp")
target_link_libraries(GloveExample)

include_directories("test/include")
add_executable(Test "test/main.cpp" "test/code.cpp")
target_link_libraries(Test)

add_subdirectory(docs)
