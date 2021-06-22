# ifndef QUADRATIC_NEURAL_NETWORK
# define QUADRATIC_NEURAL_NETWORK
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "../../shared/generate_random_arr.h"
# include "util.h"
# define INPUT_DIM 3

// A simple neural network that creates outputs
// based on a numbers of input.

// *** simple version ***////////////

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
            x[i][j]=rand_double(0,50);
        }

        // Constant
        x[i][INPUT_DIM-1]=1;
        
        // The already-defined function y = 2x^2 + 6x + 4
        y[i]= calculate_sigmoid(10*x[i][0]*x[i][0]+10*x[i][1]+4);

    }

    *(k)=x;
    *(k+1)=&y;
    return k;
}


double * quadratic_normalize_y(double *y,int arrSize){

    double * normalized_y=(double *)calloc(arrSize,sizeof(double));

    // divide the value by maximum value to normalize between [-1, 1].
    double max_value= find_max_value(y,arrSize);

    for(int i=0;i<arrSize;i++){
        normalized_y[i]=y[i]/max_value;
    }

    return normalized_y;
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
            d_err_arr_w[i]+= 
            (y[k]-calculate_sigmoid(x_dot_w))*x[k][i]*calculate_diff_sigmoid(x_dot_w);
        }

    }

    return d_err_arr_w;

}

////////////////////////////////////////////////////////
// Now calculating the weights with hidden node(s) /////
// This could be faster & better than the upper one...      /////
////////////////////////////////////////////////////////




# endif