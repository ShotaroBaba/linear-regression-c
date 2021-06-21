# ifndef SIMPLE_NEURAL_NETWORK
# define SIMPLE_NEURAL_NETWORK 
# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# ifndef GENERATE_RANDOM_ARR
# define GENERATE_RANDOM_ARR
# include "generate_random_arr.h"
# endif

// A simple neural network that creates outputs
// based on a numbers of input.

double ** simple_create_sample_input(int arrSize);
double * simple_create_random_weight(int arrSize);

// Create sample input to a single node
double ** simple_create_sample_input(int arrSize) {

    double * x = (double *)calloc(arrSize,sizeof(double));
    double * y = (double *)calloc(arrSize,sizeof(double));
    double ** k = (double **)calloc(2,sizeof(double*));

    // In this case, only one weight is created...
    for(int i=0;i<arrSize;i++){
        x[i]=i;
        y[i]=2*i;
    }

    *(k)=x;
    *(k+1)=y;
    return k;
}

// Create a random weight.
double * simple_create_random_weight(int arrSize) {

    double * r = (double *)calloc(arrSize,sizeof(double));
    for (int i=0;i<arrSize;i++) {
        r[i]=rand_double(0,0.5);
    }

    return r;
}

double * simple_multiply_weight(double * x, double w, int arrSize) {

    double * r=  (double *)calloc(arrSize,sizeof(double));
    
    for (int i=0;i<arrSize;i++){
        r[i]=x[i]*w;
    }

    return r;
}

double simple_calculate_error(double w, double * y, int arrSize){
    
    double  error=0;

    for(int i=0;i<arrSize-1;i++){
        error+=pow((y[i]-w),2);
    }

    return error;
}

double  simple_gradient_descent(double  w, double * x, 
double * y, int arrSize){

    double d_err=0;

    // In this case, only one error rate for 
    // one value is allocated.
    for (int k=0;k<arrSize;k++){
        d_err += -2*(y[k]-w*x[k])*x[k];
    }

    return d_err;

}

# endif /* SIMPLE_NEURAL_NETWORK */