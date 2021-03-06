#!/bin/bash

rm -rf tmp
mkdir tmp
cd tmp

#Eigen
git clone --depth 1 https://gitlab.com/libeigen/eigen.git
mkdir ../../DynAutoDiff/eigen3
cp -r eigen/Eigen ../../DynAutoDiff/eigen3

#boost.math
git clone --depth 1 https://github.com/boostorg/math.git
cp -r math/include/boost ../../DynAutoDiff

#boost.json
git clone --depth 1 https://github.com/boostorg/json.git
cp -r json/include/boost ../../DynAutoDiff

#boost.assert
git clone --depth 1 https://github.com/boostorg/assert.git
cp -r assert/include/boost ../../DynAutoDiff

#boost.lexical_cast
git clone --depth 1 https://github.com/boostorg/lexical_cast.git
cp -r lexical_cast/include/boost ../../DynAutoDiff

cd ..
rm -rf tmp
