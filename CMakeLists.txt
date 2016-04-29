cmake_minimum_required(VERSION 2.8)
project(vienna_benchmark)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

message("-- Build type: ${CMAKE_BUILD_TYPE}")

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffast-math -Wall")

# CUDA
find_package(CUDA)

# ViennaCL
option(BUILD_TESTING OFF)
option(ENABLE_UBLAS OFF)
option(ENABLE_OPENMP ON)
#option(ENABLE_OPENCL ON) # check if gpu found

if(${CUDA_FOUND})
    option(ENABLE_CUDA ON)
	list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
	include_directories(${CUDA_INCLUDE_DIR})
endif(${CUDA_FOUND})

include_directories(${INCLUDE_DIR} ${DEPS_DIR}/viennacl-dev)

add_executable(inner_product ${SRC_DIR}/inner_product.cpp)
add_executable(matrix_mult ${SRC_DIR}/matrix_mult.cpp)

if (${CUDA_FOUND})
	target_link_libraries(inner_product cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	target_link_libraries(matrix_mult cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif(${CUDA_FOUND})
