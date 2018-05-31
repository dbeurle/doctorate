
cmake_minimum_required(VERSION 3.9)

project(chain_length VERSION 0.1 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 14)

find_package(VTK COMPONENTS vtkIOXML REQUIRED)
find_package(Boost 1.40 COMPONENTS filesystem REQUIRED)

add_executable(chain_length main.cpp)
