cmake_minimum_required(VERSION 4.2)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(pmpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(VENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv")

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("${VENV_DIR}/include/python3.12")
set(TORCH_PATH "${VENV_DIR}/lib/python3.12/site-packages/torch/include")

include_directories("${TORCH_PATH}")
include_directories("${TORCH_PATH}/torch/csrc/api/include")

file(GLOB_RECURSE SRC_FILES *.cu *.cpp)