FROM ubuntu:18.04
ENV HOME /root

RUN apt-get update
RUN apt-get install -y git curl ninja-build zip

# install cmake
WORKDIR ${HOME}
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz
RUN tar xf cmake-3.17.3-Linux-x86_64.tar.gz
RUN cp ${HOME}/cmake-3.17.3-Linux-x86_64/bin/* /usr/bin/
RUN cp -r ${HOME}/cmake-3.17.3-Linux-x86_64/share/cmake-3.17 /usr/share/

# install host protoc
RUN curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
RUN unzip protoc-3.11.4-linux-x86_64.zip -d protoc-3.11.4-linux-x86_64
RUN cp protoc-3.11.4-linux-x86_64/bin/* /usr/bin/

# install NDK
RUN curl -L -o android-ndk-r21b-linux-x86_64.zip https://dl.google.com/android/repository/android-ndk-r21b-linux-x86_64.zip
RUN unzip android-ndk-r21b-linux-x86_64.zip
ENV NDK_ROOT ${HOME}/android-ndk-r21b

# clone mlfe library and update submodule.
RUN git clone https://github.com/shi510/mlfe
WORKDIR ${HOME}/mlfe
RUN git checkout developer_preview
RUN git submodule update --init --recursive

# build mlfe using cross compiler for aarch64 with XNNPACK.
RUN mkdir ${HOME}/mlfe/build
WORKDIR ${HOME}/mlfe/build
RUN cmake \
    -D MLFE_LITE=ON \
    -D BUILD_TEST=ON \
    -D USE_XNNPACK=ON \
    -D ANDROID_ABI=arm64-v8a \
    -D ANDROID_PLATFORM=26 \
    -D ANDROID_LINKER_FLAGS="-llog" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -D CMAKE_INSTALL_PREFIX=${HOME}/mlfe/mlfe_installed \
    -G Ninja ..
RUN ninja
