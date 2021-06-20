# ifndef SIMPLE_NEURAL_NETWORK
# define SIMPLE_NEURAL_NETWORK 
# include <stdio.h>
# include <stdlib.h>

# ifndef GENERATE_RANDOM_ARR
# define GENERATE_RANDOM_ARR
# include "generate_random_arr.h"
# endif

// A simple neural network that creates outputs
// based on a numbers of input.

double ** create_sample_input(int arrSize);
double * create_random_weight(int arrSize);

// Create sample input to a single node
double ** create_sample_input(int arrSize) {

    double * x = (double *)calloc(arrSize,sizeof(double));
    double * y = (double *)calloc(arrSize,sizeof(double));
    double ** k = (double **)calloc(2,sizeof(double*));

    for(int i=0;i<arrSize;i++){
        x[i]=i;
        y[i] = (i/10)+rand_double(0,0.1);
    }

    *(k)=x;
    *(k+1)=y;
    return k;
}

// Create a 
double * create_random_weight(int arrSize) {
    double * r = (double *)calloc(arrSize,sizeof(double));

    for (int i=0;i<arrSize;i++) {
        r[i]=rand_double(0,3.0);
    }

    return r;
}

# endif