#!/bin/bash

BUILD_TESTS=OFF
while getopts "tc" opt; do
    case $opt in
        t) BUILD_TESTS=ON ;;
        c) rm -rf build/* ;;
    esac
done

cd build/
cmake .. -DBUILD_TESTS=$BUILD_TESTS
cmake --build .