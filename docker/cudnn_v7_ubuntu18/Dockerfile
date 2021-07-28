FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV HOME /root

RUN apt-get update
RUN apt-get install -y git curl gcc-8 g++-8 ninja-build zip
RUN rm -f /usr/bin/gcc
RUN rm -f /usr/bin/g++
RUN ln -s /usr/bin/gcc-8 /usr/bin/gcc
RUN ln -s /usr/bin/g++-8 /usr/bin/g++

# install cmake
WORKDIR $HOME
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz
RUN tar xf cmake-3.17.3-Linux-x86_64.tar.gz
WORKDIR ${HOME}/cmake-3.17.3-Linux-x86_64
RUN cp ./bin/* /usr/bin/
RUN cp -r ./share/cmake-3.17 /usr/share/

# install host protoc
WORKDIR ${HOME}
RUN curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
RUN unzip protoc-3.11.4-linux-x86_64.zip -d protoc-3.11.4-linux-x86_64
RUN cp protoc-3.11.4-linux-x86_64/bin/* /usr/bin/

# clone mlfe library and update submodule.
WORKDIR $HOME
RUN git clone https://github.com/shi510/mlfe
WORKDIR ${HOME}/mlfe
RUN git checkout developer_preview
RUN git submodule update --init --recursive

# build mlfe using cross compiler for cudnn.
WORKDIR ${HOME}/mlfe
RUN mkdir build
WORKDIR ${HOME}/mlfe/build
RUN cmake \
    -D BUILD_TEST=ON \
    -D BUILD_EXAMPLE=ON \
    -D USE_CUDNN=ON \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/root/mlfe_lib \
    -G Ninja ..
RUN ninja
