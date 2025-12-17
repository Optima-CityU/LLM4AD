#!/bin/bash


cd src/cvrp-decomposition
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake
make

cd ../../../
cp src/cvrp-decomposition/build/hgs bin/hgs-TV