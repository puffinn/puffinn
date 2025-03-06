test: build
    build/Test

build:
    cmake --build build

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake ..

generate-compile-commands:
    just clean
    bear -- cmake --build build
