cmake_minimum_required(VERSION 3.5)
project(Puffinn CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -O3 -g")

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories("include")

add_executable(GloveExample "examples/glove.cpp")
target_link_libraries(GloveExample ${OpenMP_CXX_LIBRARIES})

include_directories("test/include")
add_executable(Test "test/main.cpp" "test/code.cpp")
target_link_libraries(Test ${OpenMP_CXX_LIBRARIES})

add_subdirectory(docs)
