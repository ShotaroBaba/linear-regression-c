# ifndef QUADRATIC_NEURAL_NETWORK
# define QUADRATIC_NEURAL_NETWORK
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "../../shared/generate_random_arr.h"
# include "../../shared/min_max.h"
# include "util.h"
# define HIDDEN_LAYER_SIZE 2
# define HIDDEN_LAYER_1_NODE_NUM 15
# define HIDDEN_LAYER_2_NODE_NUM 1
# define INPUT_DIM 3

// A neural network with hidden nodes that creates outputs
// based on a numbers of inputs. There are no biases.

// The sigmoid function that this program uses is 
// f(x)=1/1+exp(-x);

// *** a version with hidden layer ***////////////

double *** quadratic_create_sample_input(int arrSize, int x_dim);
double * simple_create_random_weight(int arrSize);
double *** create_output_array(int arrSize);


// Create sample input to a single node
// Note that a structure of the layer has already been 
// decided

double *** quadratic_create_sample_input(int arrSize, int x_dim) {

    // The dimension of x is 3 in this case.
    double *** k = (double ***)calloc(2,sizeof(double**));
    double ** x = (double **)calloc(arrSize,sizeof(double*));
    for (int i=0;i<arrSize;i++) {
        *(x+i)=(double *)calloc(x_dim,sizeof(double));
    }

    double * y = (double *)calloc(arrSize,sizeof(double*));
    double * random_a = (double *)calloc(x_dim,sizeof(double *));  
    
    for(int i=0;i<x_dim;i++){
        random_a[i]=rand_double(0,1);
    }


    // In this case, only one weight is created...
    for(int i=0;i<arrSize;i++){
        
        // For this function, the neural network find an optimal weights.;
        for(int j=0;j<x_dim;j++){
            y[i]= x[i][j]*random_a[j];
        }
        
    }
    
    // Normalize before inputting the values
    double max_y = findMax(y, arrSize);

    for(int i=0;i<x_dim-1;i++){
        y[i]/=max_y;
    }

    // Return all defined arrays.
    *(k)=x;
    *(k+1)=&y;

    return k;
}


// Create layer here.
// The weight & output f_w_x is created simultaneously.
double *** quadratic_create_func_output_arr(int arrSize){
    
    double *** f_w_x_arr = (double ***) calloc(arrSize, sizeof(double**));
    
    for(int i=0;i<arrSize;i++){
        *(f_w_x_arr+i)= (double **) calloc(HIDDEN_LAYER_SIZE, sizeof(double*));

        f_w_x_arr[i][0] = (double *) calloc(HIDDEN_LAYER_1_NODE_NUM,sizeof(double));
        f_w_x_arr[i][1] = (double *) calloc(HIDDEN_LAYER_2_NODE_NUM,sizeof(double));	 

    }

    return f_w_x_arr;
}

double *** quadratic_create_random_sample_weight() {


    // All nodes are treated with sigmoid function.
    double *** w = (double ***) calloc(HIDDEN_LAYER_SIZE,sizeof(double**));

    // Layer 1 weight
    *(w+0) = (double **) calloc(HIDDEN_LAYER_1_NODE_NUM,sizeof(double*));
    
    // Layer 2 weight
    *(w+1) = (double **) calloc(HIDDEN_LAYER_2_NODE_NUM,sizeof(double*));
    
    // Layer 1, weight w_0_0,w_0_1,w_0_2　for x = [x_1,x_2,x_3];
    for(int i=0;i<HIDDEN_LAYER_1_NODE_NUM;i++){
        w[0][i]=calloc(INPUT_DIM,sizeof(double));
        for(int j=0;j<INPUT_DIM;j++){
            w[0][i][j]=rand_double(0,1);
        }
    }
    
    // Move on to the next layer...
    // Layer 2, weight w_1_1, for a = [ a_0_0 ];
    for (int i=0;i<HIDDEN_LAYER_2_NODE_NUM;i++){
        w[1][i]=calloc(HIDDEN_LAYER_1_NODE_NUM,sizeof(double));
        for(int j=0;j<HIDDEN_LAYER_1_NODE_NUM;j++){
            w[1][i][j]=rand_double(0,1);
        }
    }

    return w;
}

double * quadratic_calculate_error(double * w, double ** x, 
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
            (y[k]-calculate_sigmoid(x_dot_w))*calculate_diff_sigmoid(x_dot_w);
        }

    }

    return d_err_arr_w;

}

# endif