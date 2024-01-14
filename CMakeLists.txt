cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(kernel LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})

add_library(${PROJECT_NAME} SHARED kernel.cu)
target_include_directories(${PROJECT_NAME} PUBLIC $ENV{CUDA_SAMPLES})
