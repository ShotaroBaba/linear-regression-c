// Define default values
#define MAX_ARR_SIZE 100000
#define RAND_ERR_MIN -0.5
#define RAND_ERR_MAX 0.5
#define NUM_VALUES 3
#define MIN_a_RANGE 1
#define MIN_b_RANGE 1
#define MAX_a_RANGE 10
#define MAX_b_RANGE 10
//////////////////////////

# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <math.h>


# include "../shared/average_func.h"
# include "../shared/generate_random_arr.h"
# include "lib/quadratic_neural_network.h"

// Declaring functions ///////////
double ** generate_random_data();
double rand_double(double a, double b);
double * create_array();
double calculate_average(double *a, int arrSize);
double calculate_uncorrelated_std(double *a, double a_avg,int arrSize);
double calculate_sample_correlation_coefficient(double *a, double *b,
double avg_x, double avg_y,
double u_std_a, double u_std_b,
int arrSize);
//////////////////////////////////

// Simple linear regression
int main(int argc, char ** argv) {
    
    
    double *** w = quadratic_create_random_sample_weight();
    double *** initial_input = quadratic_create_sample_input(100);

    double ** x = initial_input[0];
    double ** f_w_x = initial_input[1];
    double * y = initial_input[2][0];


    ///////////////////////////////////////
    // Start Forward Propagation //////////
    ///////////////////////////////////////
    printf("Start Forward Propagation (For layer 1)\n");
    // For layer 1.
    for(int i =0;i<HIDDEN_LAYER_1_NODE_NUM;i++){
        
        f_w_x[0][i]=0;

        for(int k=0;k<INPUT_DIM;k++){
            printf("Calculating w[0][%d][%d]\n", i, k);
            f_w_x[0][i]+=w[0][i][k]*x[i][k];
        }

        f_w_x[0][i]=calculate_sigmoid(f_w_x[0][i]);
    }

    // For layer 2.
    // Note: The output of f_w_x can be multi-dimensional.
     printf("Start Forward Propagation (For layer 2)\n");
    for (int i = 0;i<HIDDEN_LAYER_2_NODE_NUM;i++){
        
        f_w_x[1][i]=0;
        for(int k=0;k<HIDDEN_LAYER_1_NODE_NUM;k++){
                            // Can be f_w_x[0][j][k]...
            printf("Calculating w[1][%d][%d]\n", i, k);
            f_w_x[1][i]+=w[1][i][k]*f_w_x[0][k];
        }

        f_w_x[1][i]=calculate_sigmoid(f_w_x[0][i]);
    }
    ///////////////////////////////////////
    // Forward Propagation End ////////////
    ///////////////////////////////////////

    // Start back propagation.

}

