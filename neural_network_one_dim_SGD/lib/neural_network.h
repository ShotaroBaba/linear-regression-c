# ifndef NEURAL_NETWORK_ONE_DIM
# define NEURAL_NETWORK_ONE_DIM
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "../../shared/generate_random_arr.h"
# include "../../shared/min_max.h"
# include "util.h"
# define HIDDEN_LAYER_SIZE 2
# define INPUT_DIM 3

// A neural network with hidden nodes that creates outputs
// based on a numbers of inputs. There are no biases.

// The sigmoid function that this program uses is 
// f(x)=1/1+exp(-x);

// *** a version with hidden layer ***////////////

double *** quadratic_create_sample_input(int arrSize, int x_dim);
double *** create_func_output_arr(int arrSize,int layer_num, int* node_num);
double *** create_random_weight(int layer_size,int* node_num);
double *** create_test_data_set(double * a, int arrSize, int x_dim);
// Create sample input to a single node
// Note that a structure of the layer has already been 
// decided

double *** quadratic_create_sample_input(int arrSize, int x_dim) {

    // The dimension of x is 3 in this case.
    double *** k = (double ***)calloc(3,sizeof(double**));
    double ** x = (double **)calloc(arrSize,sizeof(double*));
    for (int i=0;i<arrSize;i++) {
        *(x+i)=(double *)calloc(x_dim,sizeof(double));
    }

    double * y = (double *)calloc(arrSize,sizeof(double));
    double * random_a = (double *)calloc(x_dim,sizeof(double *));  
    
    for(int i=0;i<x_dim;i++){
        random_a[i]=rand_double(0,3.0);
    }


    // In this case, only one weight is created...
    for(int i=0;i<arrSize;i++){
        y[i] = 0;
        // For this function, the neural network find an optimal weights.;
        for(int j=0;j<x_dim;j++){
            x[i][j]=rand_double(0,10);
            y[i]+= pow(x[i][j],j+1);
        }

    }



    // Normalize before inputting the values
    double max_y = findMax(y, arrSize);

    for(int i=0;i<arrSize;i++){
        y[i]/=max_y;
    }

    // Return all defined arrays.
    *(k)=x;
    *(k+1)=&y;
    *(k+2)=&random_a;

    return k;
}

double *** create_test_data_set(double * a,int arrSize,int x_dim){
    
    double *** k = (double ***)calloc(2,sizeof(double**));
    double ** x = (double **)calloc(arrSize,sizeof(double*));
    for (int i=0;i<arrSize;i++) {
        *(x+i)=(double *)calloc(x_dim,sizeof(double));
    }

    double * y = (double *)calloc(arrSize,sizeof(double));  


    // In this case, only one weight is created...
    for(int i=0;i<arrSize;i++){
        y[i] = 0;
        // For this function, the neural network find an optimal weights.;
        for(int j=0;j<x_dim;j++){
            x[i][j]=rand_double(0,10);
            y[i]+= pow(x[i][j],j+1);
        }

    }

    // Normalize before inputting the values
    double max_y = findMax(y, arrSize);

    for(int i=0;i<arrSize;i++){
        y[i]/=max_y;
    }

    // Return all defined arrays.
    *(k)=x;
    *(k+1)=&y;

    return k;
}

// Note, the first node dim needs to be the same as the one of the input num.
double *** create_func_output_arr(int arrSize,int layer_num, int* node_num){

    // Size of outputs per training data size.
    double *** f_w_x_arr = (double ***) calloc(arrSize, sizeof(double**));

    for(int i=0;i<arrSize;i++){
        
        *(f_w_x_arr+i)= (double **) calloc(layer_num, sizeof(double*));

        for(int j=0;j<layer_num-1;j++){
            f_w_x_arr[i][j]= (double *) calloc(node_num[j+1], sizeof(double));
        }
    }


    return f_w_x_arr;
}

// Create array test
double *** create_func_output_arr_test(int arrSize,int layer_num, double ** x, int* node_num){

    // Size of outputs per training data size.
    double *** f_w_x_arr = (double ***) calloc(arrSize, sizeof(double**)); 

    // f_w_x [0]
    

    // f_w_x [n] n > 0
    for(int i=0;i<arrSize;i++){    
        *(f_w_x_arr+i)= (double **) calloc(layer_num, sizeof(double*));

        for(int j=0;j<layer_num;j++){
            f_w_x_arr[i][j]= (double *) calloc(node_num[j], sizeof(double));
        }
    }

    for(int i = 0;i<arrSize;i++){
        for(int j=0;j<node_num[0];j++){
            f_w_x_arr[i][0][j]=x[i][j];    
        }
    }

    return f_w_x_arr;
}

double *** create_func_output_arr_test_avg(int arrSize,int layer_num, int* node_num) {
    // Size of outputs per training data size.
    double *** f_w_x_arr = (double ***) calloc(arrSize, sizeof(double**)); 

    for(int i=0;i<arrSize;i++){    
        *(f_w_x_arr+i)= (double **) calloc(layer_num, sizeof(double*));

        for(int j=0;j<layer_num;j++){
            f_w_x_arr[i][j]= (double *) calloc(node_num[j], sizeof(double));
        }
    }

     return f_w_x_arr;
}


// The array node_num includes the dimension of the inpu
double *** create_random_weight(int layer_size, int* node_num){
    
    // All nodes are treated with sigmoid function.
    double *** w = (double ***) calloc(layer_size,sizeof(double**));

    for(int i=0; i<layer_size-1;i++){
        
        w[i] = (double **) calloc(node_num[i+1], sizeof(double*));

        // Layer i + 1 (next layer)
        for(int j=0; j< node_num[i+1];j++){


            // Layer i (current layer)
            w[i][j]=(double * ) calloc(node_num[i],sizeof(double));
            for(int k=0;k<node_num[i];k++){
                w[i][j][k]=rand_double(-0.5,0.5);
            }
        }
    }

    return w;
}

// Xavier weight initialization
double *** create_random_weight_xavier(int layer_size, int* node_num){
    // All nodes are treated with sigmoid function.
    double *** w = (double ***) calloc(layer_size,sizeof(double**));

    for(int i=0; i<layer_size-1;i++){
        
        w[i] = (double **) calloc(node_num[i+1], sizeof(double*));

        // Layer i + 1 (next layer)
        for(int j=0; j< node_num[i+1];j++){
            // Layer i (current layer)
            w[i][j]=(double * ) calloc(node_num[i],sizeof(double));
            for(int k=0;k<node_num[i];k++){
                w[i][j][k]=rand_double(-sqrt(6)/(node_num[i+1]+node_num[i]),sqrt(6)/(node_num[i+1]+node_num[i]));
            }
        }
    }

    return w;

}

# endif