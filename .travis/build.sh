#!/bin/bash
set -ex

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

mkdir build
cd build

CMAKE_ARGS=('-DBUILD_TEST=ON')
CMAKE_ARGS+=('-DCMAKE_INSTALL_PREFIX=../install')
cmake .. ${CMAKE_ARGS[*]}
make "-j$(nproc)" install
# if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
#     make "-j$(nproc)" install
# fi
