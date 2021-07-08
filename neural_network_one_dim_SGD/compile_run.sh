#!/bin/bash

gcc -o test.out main.c -g -L ./lib  -L ./../../shared -l m  && ./test.out