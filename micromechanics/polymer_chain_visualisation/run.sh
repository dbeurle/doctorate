#!/bin/bash

clang++ -O3 -std=c++1z -I/usr/include/vtk NetworkGenerator.cpp -L/usr/lib64/vtk -lvtkCommonDataModel -lvtkCommonCore -lvtkIOXML -o network_generator
./network_generator
