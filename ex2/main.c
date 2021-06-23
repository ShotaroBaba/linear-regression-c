// Define default values
#define MAX_ARR_SIZE NUM_TRAIN_SIZE000
#define RAND_ERR_MIN -0.5
#define RAND_ERR_MAX 0.5
#define NUM_VALUES 3
#define MIN_a_RANGE 1
#define MIN_b_RANGE 1
#define MAX_a_RANGE 10
#define MAX_b_RANGE 10
#define NUM_TRAIN_SIZE 1000
#define NUM_TEST_SIZE 1000
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
    
    double alpha = 0.3;
    double *** w = quadratic_create_random_sample_weight();
    double *** initial_input = quadratic_create_sample_input(NUM_TRAIN_SIZE);

    double ** x = initial_input[0];
    double * y = initial_input[1][0];
    double *** f_w_x = quadratic_create_func_output_arr(NUM_TRAIN_SIZE);
    double *** d_err = quadratic_create_func_output_arr(NUM_TRAIN_SIZE);
    int epoch = 5000;


    printf("Start Forward Propagation (For layer 1)\n");
    // For layer 1.
    for(int _ = 0; _ < epoch;_++){
        double mean_square_error = 0;
        for(int train_index=0;train_index<NUM_TRAIN_SIZE;train_index++){

            ///////////////////////////////////////
            // Start Forward Propagation //////////
            ///////////////////////////////////////
            for(int i =0;i<HIDDEN_LAYER_1_NODE_NUM;i++){
                
                f_w_x[train_index][0][i]=0;

                for(int k=0;k<INPUT_DIM;k++){
                    // printf("Calculating w[0][%d][%d]\n", i, k);
                    f_w_x[train_index][0][i]+=w[0][i][k]*x[train_index][k];
                }

                f_w_x[train_index][0][i]=calculate_sigmoid(f_w_x[train_index][0][i]);
            }

            // For layer 2.
            // Note: The output of f_w_x can be multi-dimensional.
            // printf("Start Forward Propagation (For layer 2)\n");
            for (int i = 0;i<HIDDEN_LAYER_2_NODE_NUM;i++){
                
                f_w_x[train_index][1][i]=0;
                for(int k=0;k<HIDDEN_LAYER_1_NODE_NUM;k++){
                    // printf("Calculating w[1][%d][%d]\n", i, k);
                    f_w_x[train_index][1][i]+=w[1][i][k]*f_w_x[train_index][0][k];
                }

                f_w_x[train_index][1][i]=calculate_sigmoid(f_w_x[train_index][1][i]);
            }
            
            ///////////////////////////////////////
            // Forward Propagation End ////////////
            ///////////////////////////////////////

            ///////////////////////////////////////
            /// Now calculate output error/////////
            ///////////////////////////////////////

            // Do this on Layer 3 first.
            for(int i = 0;i<HIDDEN_LAYER_2_NODE_NUM;i++){
                d_err[train_index][1][i]=0;
                double tmp_sum_backword=0;
            
                for(int k=0;k<HIDDEN_LAYER_1_NODE_NUM;k++){
                    tmp_sum_backword+=w[1][i][k]*f_w_x[train_index][0][k];
                }

                for(int k=0;k<HIDDEN_LAYER_1_NODE_NUM;k++){
                    d_err[train_index][1][i] += (f_w_x[train_index][1][i] - y[train_index])*
                    calculate_diff_sigmoid(tmp_sum_backword);
                }

                mean_square_error+=pow(f_w_x[train_index][1][i] - y[train_index],2)/2;
            }
            
            ///////////////////////////////////////
            /// Output error calculation finished /
            ///////////////////////////////////////

            ///////////////////////////////////////
            // Start Backward Propagation //////////
            ///////////////////////////////////////

            // Now on Layer 2.
            for(int i = 0;i<HIDDEN_LAYER_1_NODE_NUM;i++){
                double tmp_sum_backword = 0;

                for(int k=0;k<INPUT_DIM;k++){
                    tmp_sum_backword+=w[0][i][k]*x[train_index][k];
                }

                d_err[train_index][0][i]=0;
                
                for(int k=0;k<HIDDEN_LAYER_2_NODE_NUM;k++){
                    d_err[train_index][0][i] += w[1][k][i]*d_err[train_index][1][k]*
                    calculate_diff_sigmoid(tmp_sum_backword);
                }

            }

            ////////////////////////////////////////
            // Backword Propagation End ////////////
            ///////////////////////////////////////
        

            ///////////////////////////////////////
            // Gradient Descent ///////////////////
            ///////////////////////////////////////

            // Layer 3/////////////////////////////
            for(int i = 0;i<HIDDEN_LAYER_2_NODE_NUM;i++){
                for(int k=0;k<HIDDEN_LAYER_1_NODE_NUM;k++){
                    w[1][i][k]-=alpha*d_err[train_index][1][i]*f_w_x[train_index][0][k]/NUM_TRAIN_SIZE;
                }
            }

            // Layer 2///////////////////////////////
            for(int i =0;i<HIDDEN_LAYER_1_NODE_NUM;i++){
                for (int k=0;k<INPUT_DIM;k++){
                    w[0][i][k]-=alpha*d_err[train_index][0][i]*f_w_x[train_index][0][k]/NUM_TRAIN_SIZE;
                }
            }
            
            ///////////////////////////////////////
            // Gradient descent end ///////////////
            ///////////////////////////////////////
            // Update all weight based on the calculated errors.

        }

        printf("Epoch %d, Mean square error: %f\n", _, mean_square_error);
    }


}
