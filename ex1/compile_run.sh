#!/bin/bash

gcc -o test.out main.c -L ./lib  -L ./../../shared -l m  && ./test.out