#!/bin/zsh
cd Build && {
  mkdir -p Release && {
    cd Release
    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS=$1 ../..
    cmake --build . -j
  }
}