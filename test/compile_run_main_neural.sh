#!/bin/bash

(cd ../neural_network_one_dim_SGD && \
gcc -o test_small.out main_test.c -g -L ./lib  -L ./../../shared -l m  && \
 ./test_small.out)