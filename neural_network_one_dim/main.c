// Define default values
#define MAX_ARR_SIZE 
#define RAND_ERR_MIN -0.5
#define RAND_ERR_MAX 0.5
#define NUM_VALUES 3
#define X_DIM 10
#define Y_DIM 5
#define MIN_a_RANGE 1
#define MIN_b_RANGE 1
#define MAX_a_RANGE 10
#define MAX_b_RANGE 10
#define NUM_TRAIN_SIZE 20000
#define NUM_TEST_SIZE 1000
#define NUM_EPOCHS 500
//////////////////////////

# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <math.h>


# include "../shared/average_func.h"
# include "../shared/generate_random_arr.h"
# include "lib/neural_network.h"

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
    
    time_t t;

    srand((unsigned)time(&t));
    int x_dim = 10;
    int y_dim = 1;
    int layer_num = 6;

    // Note: The array size of node num must be more than 1.
    // In addtion, the last dimension must be 1.
    int node_num[] = {x_dim, 4,3,7,3,y_dim};
    int node_size = sizeof(node_num)/sizeof(int);

    printf("Node size: %d\n", node_size);

    if(node_size != layer_num) {
        printf("Error: the number of layer and the node size does not match.\n");
        return 1;
    }

    double alpha = 0.1;
    double *** initial_input = quadratic_create_sample_input(NUM_TRAIN_SIZE,x_dim);
    double ** x  = initial_input[0];
    double * y = initial_input[1][0];

    double *** w = create_random_weight(layer_num,node_num);
    double *** f_w_x = create_func_output_arr(NUM_TRAIN_SIZE,layer_num,node_num);
    double *** d_err = create_func_output_arr(NUM_TRAIN_SIZE,layer_num,node_num);

    ////////////////////////////////////////////////////////////////
    // Forward Propagation ///////////
    ////////////////////////////////////////////////////////////////
    
    // Start the computation with
    // Initialize mean square_error.
    for(int epoch = 0; epoch < NUM_EPOCHS; ++epoch){
        double mean_square_error = 0;
        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
            for(int j = 0;j<layer_num-1;j++){
                for(int k =0;k<node_num[j+1];k++){
                    // If a layer is zero, then input a first num
                    if(j==0) {
                        double * x_dot_w = calculate_dot(w[j][k], x[train_index],x_dim);
                        f_w_x[train_index][j][k] = calculate_sigmoid_arr(x_dot_w,node_num[j]);
                        free(x_dot_w);
                    }
                    else {
                        double * x_dot_w = calculate_dot(w[j][k], f_w_x[train_index][j-1],node_num[j]);
                        f_w_x[train_index][j][k] = calculate_sigmoid_arr(x_dot_w,node_num[j]);
                        free(x_dot_w);
                    }
                }
            }
        }

        // for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
        //     for(int j = 0;j<layer_num-1;j++){
        //         for(int k=0;k<node_num[j+1];k++){
        //             printf("i: %d, j: %d, k: %d, f_w_x: %f\n", train_index,j,k,f_w_x[train_index][j][k]);
        //             }

        //     }   
        // }

        ////////////////////////////////////////////////////////////////
        // Forward Propagation End ///////////
        ////////////////////////////////////////////////////////////////
        
        ////////////////////////////////////////////////////////////////
        // Backward Propagation ///////////
        ////////////////////////////////////////////////////////////////
        
        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){

            // Backpropagate per layer.
            for(int j = layer_num-1;j>0;j--){
                
                for(int k =0;k<node_num[j];k++){
                    d_err[train_index][j-1][k]=0;

                    // TODO: The function can be created in more generic way.
                    // If a layer is zero, then input a first num
                    if(j==layer_num-1 && j>1) {
                                    
                        // Calcualte previous layer dot product first.
                        double * x_dot_w = calculate_dot(w[j-1][k], f_w_x[train_index][j-2],node_num[j-1]);

                        for(int l = 0;l<node_num[j];l++) {
                            // The value l can be fitted into y...
                            d_err[train_index][j-1][k] += (y[train_index] - f_w_x[train_index][j-1][k])
                            *calculate_sigmoid_diff_arr(x_dot_w,node_num[j-1]);
                        }
                        
                        mean_square_error+=pow(d_err[train_index][j-1][k],2)/(2*NUM_TRAIN_SIZE);
                        free(x_dot_w);
                    
                    }
                    else if(j==1) {
                        double * x_dot_w = calculate_dot(w[j-1][k], x[train_index],node_num[j-1]);
                        for(int l = 0;l<node_num[j+1];l++){
                            d_err[train_index][j-1][k] += w[j][l][k]*d_err[train_index][j][l]*
                            calculate_sigmoid_diff_arr(x_dot_w,node_num[j-1]);
                            
                        }
                        free(x_dot_w);

                    }
                    else {
                        double * x_dot_w = calculate_dot(w[j-1][k], f_w_x[train_index][j-2],node_num[j-1]);
                        for(int l = 0;l<node_num[j+1];l++){
                            d_err[train_index][j-1][k] += w[j][l][k]*d_err[train_index][j][l]*
                            calculate_sigmoid_diff_arr(x_dot_w,node_num[j-1]);   
                        }
                        free(x_dot_w);

                    }
                }
            }

        }
        ////////////////////////////////////////////////////////////////
        // Backward Propagation End ///////////
        ////////////////////////////////////////////////////////////////
        


        ////////////////////////////////////////////////////////////////
        // Gradient Descent　//////////////
        ////////////////////////////////////////////////////////////////
        

        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
            for(int i =0;i<layer_num-1;i++){
                for(int j = 0;j<node_num[i+1];j++){
                    for(int k=0;k<node_num[i];k++){
                        w[i][j][k]-=alpha*d_err[train_index][i][j]*f_w_x[train_index][i][j]/NUM_TRAIN_SIZE;
                    }
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        // Gradient Descent End ///////////
        ////////////////////////////////////////////////////////////////
        
        printf("Epoch %d, MSE: %f\n", epoch, mean_square_error);
        // for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
        //     for(int j = 0;j<layer_num-1;j++){
        //         for(int k=0;k<node_num[j+1];k++){
        //             printf("i: %d, j: %d, k: %d, d_err: %f\n", train_index,j,k,d_err[train_index][j][k]);
        //         }   
        //     }
        // }
        }


}
