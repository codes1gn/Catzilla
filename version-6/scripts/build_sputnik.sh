#!/bin/bash
cd third_party/sputnik && {
  mkdir -p build && cd build && {
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS=$1 && make -j
  }
}