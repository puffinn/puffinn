cmake_minimum_required(VERSION 3.5)
project(Puffinn CXX C)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -O3 -g")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (OpenMP_CXX_INCLUDE_DIR)
        include_directories("${OpenMP_CXX_INCLUDE_DIR}")
    endif()
endif()

find_package(HDF5)
if (HDF5_FOUND)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HDF5_CXX_FLAGS}")
endif()
add_subdirectory("external/HighFive")

include_directories("include")

add_executable(GloveExample "examples/glove.cpp")
if (OpenMP_FOUND)
    target_link_libraries(GloveExample ${OpenMP_CXX_LIBRARIES})
endif()

add_executable(PuffinnJoin "join-experiments/puffinn-join.cpp")
if (OpenMP_FOUND)
    target_link_libraries(PuffinnJoin ${OpenMP_CXX_LIBRARIES} HighFive)
endif()

add_executable(StoreJaccard "join-experiments/StoreJaccard.cpp")
if (OpenMP_FOUND)
    target_link_libraries(StoreJaccard ${OpenMP_CXX_LIBRARIES} HighFive)
endif()

add_executable(XiaoEtAl "join-experiments/XiaoEtAl.cpp")
if (OpenMP_FOUND)
    target_link_libraries(XiaoEtAl ${OpenMP_CXX_LIBRARIES} HighFive)
endif()

add_executable(LSBTree "join-experiments/lsb_tree.cpp")
if (OpenMP_FOUND)
    target_link_libraries(LSBTree ${OpenMP_CXX_LIBRARIES} HighFive)
endif()


include_directories("test/include")
add_executable(Test "test/main.cpp" "test/code.cpp")
if (OpenMP_FOUND)
    target_link_libraries(Test ${OpenMP_CXX_LIBRARIES})
endif()
 
# Benchmark code
include_directories("bench")
add_library(nanobench "bench/nanobench.cpp")

add_executable(Bench "bench/bench.cpp")
target_link_libraries(Bench nanobench)
if (OpenMP_FOUND)
    target_link_libraries(Bench ${OpenMP_CXX_LIBRARIES} HighFive)
endif()

add_executable(SortBench "bench/sorthash_bench.cpp")
target_link_libraries(SortBench nanobench)

add_subdirectory(docs)
