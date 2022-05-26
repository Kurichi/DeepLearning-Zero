cmake_minimum_required(VERSION 3.14)
project(DeepLearning-Zero)

set(CMAKE_CXX_STANDARD 20)

add_executable(DeepLearning-Zero main.cpp)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(DeepLearning-Zero PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(DeepLearning-Zero Python3::Python Python3::NumPy)