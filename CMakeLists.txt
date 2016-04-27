cmake_minimum_required(VERSION 2.8)
project(vienna_benchmark)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

# CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -03 -ffast-math -Wall")
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# ViennaCL
option(BUILD_TESTING OFF)
option(ENABLE_UBLAS OFF)
option(ENABLE_OPENMP ON)
#option(ENABLE_OPENCL ON) # check if gpu found

if(${CUDA_FOUND})
    option(ENABLE_CUDA ON)
endif(${CUDA_FOUND})

include_directories(${CUDA_INCLUDE_DIRS}
                    ${INCLUDE_DIR}
                    ${DEPS_DIR}/hayai/src
                    ${DEPS_DIR}/viennacl-dev/viennacl)

set(SOURCE_FILES ${SRC_DIR}/main.cpp)
add_executable(vienna_benchmark ${SOURCE_FILES})
target_link_libraries(vienna_benchmark ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}})