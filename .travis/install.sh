#!/bin/bash
set -ex

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"
cd build

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    make install
fi
