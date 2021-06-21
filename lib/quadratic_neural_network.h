# ifndef QUADRATIC_NEURAL_NETWORK
# define QUADRATIC_NEURAL_NETWORK
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "generate_random_arr.h"
# define INPUT_DIM 3

// A simple neural network that creates outputs
// based on a numbers of input.

double *** quadratic_create_sample_input(int arrSize);
double * simple_create_random_weight(int arrSize);

// Create sample input to a single node
double *** quadratic_create_sample_input(int arrSize) {

    double ** x = (double **)calloc(arrSize,sizeof(double*));
    for (int i=0;i<arrSize;i++) {
        *(x+i)=(double *)calloc(INPUT_DIM,sizeof(INPUT_DIM));
    }

    double * y = (double *)calloc(arrSize,sizeof(double*));

    double *** k = (double ***)calloc(2,sizeof(double**));

    // In this case, only one weight is created...
    for(int i=0;i<arrSize;i++){

        for(int j=0;j<INPUT_DIM-1;j++){
            x[i][j]=rand_double(0,10);
        }

        // Constant
        x[i][INPUT_DIM-1]=1;
        
        // The already-defined function y = 2x^2 + 6x + 4
        y[i]=2*x[i][0]*x[i][0]+3*x[i][1]+4*x[i][2];
    
    }

    *(k)=x;
    *(k+1)=&y;
    return k;
}


double * quadratic_create_random_sample_weight() {

    double * w = (double *)calloc(INPUT_DIM,sizeof(double*));

    for(int i=0;i<INPUT_DIM;i++){
        w[i]=rand_double(0,3.5);
    }

    return w;
}

double *  quadratic_calculate_error(double * w, double ** x, 
double * y, int arrSize){

    double * d_err_arr_w=calloc(INPUT_DIM,sizeof(double));

    // Initialiize d_err_arr_w;
    for(int i=0;i<INPUT_DIM;i++){
        d_err_arr_w[i]=0;
    }

    // In this case, only one error rate for 
    // one value is allocated.
    for (int k=0;k<arrSize;k++){

        double x_dot_w = 0;
        for(int i=0;i<INPUT_DIM;i++){
            x_dot_w+=w[i]*x[k][i];
        }

        for(int i=0;i<INPUT_DIM;i++){
            d_err_arr_w[i] += (y[k]-x_dot_w)*x[k][i];
        }

    }

    return d_err_arr_w;

}

# endif