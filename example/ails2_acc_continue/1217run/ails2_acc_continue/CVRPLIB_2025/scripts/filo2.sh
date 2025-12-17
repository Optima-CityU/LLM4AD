#!/bin/bash


cd src/filo2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=1
make -j

cd ../../../
cp src/filo2/build/filo2 bin/filo2