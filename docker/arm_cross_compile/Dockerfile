FROM ubuntu:18.04
ENV HOME /root

RUN apt-get update
RUN apt-get install -y git curl gcc-8 g++-8 gcc-8-aarch64-linux-gnu g++-8-aarch64-linux-gnu ninja-build
RUN ln -s /usr/bin/gcc-8 /usr/bin/gcc
RUN ln -s /usr/bin/g++-8 /usr/bin/g++

# install cmake
WORKDIR $HOME
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz
RUN tar xf cmake-3.17.3-Linux-x86_64.tar.gz
WORKDIR ./cmake-3.17.3-Linux-x86_64
RUN cp ./bin/* /usr/bin/
RUN cp -r ./share/cmake-3.17 /usr/share/

# clone mlfe library and update submodule.
WORKDIR $HOME
RUN git clone https://github.com/shi510/mlfe
WORKDIR ./mlfe
RUN git checkout developer_preview
RUN git submodule update --init --recursive

# build and install protobuf using host compiler before corss compile.
# this is necessary to generate proto-cpp files.
WORKDIR ./third_party/protobuf/cmake
RUN mkdir build
WORKDIR ./build
RUN cmake -D CMAKE_BUILD_TYPE=Release -G Ninja ..
RUN ninja && ninja install

# build mlfe using cross compiler for aarch64 with XNNPACK.
WORKDIR ${HOME}/mlfe
RUN mkdir build
WORKDIR ./build
RUN cmake \
    -D USE_XNNPACK=ON \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_TOOLCHAIN_FILE=../cmake/tool_chain/arm_cross.cmake \
    -G Ninja ..
RUN ninja