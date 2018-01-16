
#!/bin/bash

clang++ -O3 -march=native -std=c++14 PolymerChainLength.cpp -o chains && ./chains
python3 Postprocessor.py
