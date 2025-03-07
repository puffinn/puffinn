install-python:
    pip uninstall puffinn
    python setup.py install

build-python:
    python setup.py build

test: build
    build/Test

build:
    cmake --build build --config Debug

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake ..

generate-compile-commands:
    just clean
    bear -- cmake --build build
