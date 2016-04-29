#!/bin/bash

cd ${HOME}

sudo apt-get install git cmake gcc g++ curl
curl -O http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda

echo "export CUDA_HOME=/usr/local/cuda-7.5" >> ${HOME}/.bashrc
echo "export LD_LIBRARY_PATH=${CUDA_HOME}"  >> ${HOME}/.bashrc
echo "PATH=${CUDA_HOME}=/bin:${PATH}"       >> ${HOME}/.bashrc
echo "export PATH"                          >> ${HOME}/.bashrc

source ${HOME}/.bashrc

git clone --recursive git://github.com/romanlarionov/viennacl-benchmark.git
cd viennacl-benchmark
mkdir build
cd build
cmake ..
make

./inner_product 1 90000000
./matrix_mult 1
