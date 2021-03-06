// NOTE: There are no optimizers for this implementation.

// Define default values
#define MAX_ARR_SIZE 
#define X_DIM 10
#define Y_DIM 5
#define BATCH_SIZE 100

#define NUM_TRAIN_SIZE 2000
#define NUM_TEST_SIZE 2000
#define NUM_EPOCHS 200
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
//////////////////////////////////


// Simple linear regression
int main(int argc, char ** argv) {
    
    time_t t;

    // Set random seed using current time.
    srand((unsigned)time(&t));
    
    int x_dim = 2;
    int y_dim = 1;
    int layer_num = 5;

    // Note: The array size of node num must be more than 1.
    // In addtion, the last dimension must be 1.
    int node_num[] = {x_dim, 10,5,2,y_dim};
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
    double * a = initial_input[2][0];
    
    // w [layer] [next_node_num] [previous_node_num]
    // f_w_x [layer] [next_node_num] 
    double *** w = create_random_weight(layer_num,node_num);
    double *** f_w_x = create_func_output_arr(NUM_TRAIN_SIZE,layer_num,node_num);
    double *** d_err = create_func_output_arr(NUM_TRAIN_SIZE,layer_num,node_num);
    double *** f_w_x_test = create_func_output_arr_test(NUM_TRAIN_SIZE,layer_num,x,node_num);

    int num_batch = NUM_TRAIN_SIZE/BATCH_SIZE;
    
    // Calculate the average of the d_err size.
    double *** d_err_average =  create_func_output_arr(num_batch,layer_num,node_num);
    double *** f_w_x_average =  create_func_output_arr(num_batch,layer_num,node_num);
    double *** f_w_x_average_test  = create_func_output_arr_test_avg(num_batch,layer_num,node_num);
    double ** x_average = (double **)calloc(num_batch,sizeof(double*));

    for (int i=0;i<num_batch;i++) {
        *(x_average+i)=(double *) calloc(x_dim,sizeof(double));
    }

    printf("%f\n", f_w_x_test[0][0][0]);
    double *bias = calloc(layer_num-1,sizeof(double));
    for (int i=0;i<layer_num-1;i++){
        bias[i]=0;
    }
    ////////////////////////////////////////////////////////////////
    // Forward Propagation ///////////
    ////////////////////////////////////////////////////////////////
    // Start the computation with
    // Initialize mean square_error.
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){
        double mean_square_error = 0;

        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
            for(int j = 1;j<layer_num;j++){
                for(int k =0;k<node_num[j];k++){
                    // If a layer is zero, then input a first num
                    double * x_dot_w = calculate_dot(w[j-1][k], f_w_x_test[train_index][j-1],node_num[j-1]);
                    f_w_x_test[train_index][j][k] = calculate_linear_arr(x_dot_w,bias[j],node_num[j]);
                    free(x_dot_w);
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        // Forward Propagation End ///////////
        ////////////////////////////////////////////////////////////////
        
        ////////////////////////////////////////////////////////////////
        // Backward Propagation ///////////
        ////////////////////////////////////////////////////////////////
        

        /// Firstly, calculate a first error.
        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
            // Calcualte previous layer dot product first.
            
            for(int k =0;k<node_num[layer_num-1];k++){
                d_err[train_index][layer_num-2][k]=0;

                double * x_dot_w = calculate_dot(w[layer_num-2][k], f_w_x_test[train_index][layer_num-2],node_num[layer_num-2]);
            
                for(int l = 0;l<node_num[layer_num-1];l++) {
                    // The value l can be fitted into y...
                    d_err[train_index][layer_num-2][k] += (f_w_x_test[train_index][layer_num-1][k]-y[train_index])
                    *1;
                }
                mean_square_error+=pow(y[train_index] - f_w_x_test[train_index][layer_num-1][k],2)/(2*NUM_TRAIN_SIZE);
                free(x_dot_w);
            }
        }

        for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
            for(int j = layer_num-3;j>-1;j--){
                for(int k =0;k<node_num[j+1];k++){
                    d_err[train_index][j][k]=0;
                    double * x_dot_w = calculate_dot(w[j][k], f_w_x_test[train_index][j],node_num[j]);
                    for(int l = 0;l<node_num[j+2];l++){
                        d_err[train_index][j][k] += w[j+1][l][k]*d_err[train_index][j+1][l]*1;       
                    }
                    free(x_dot_w);
                }
            }
        }
        
        ////////////////////////////////////////////////////////////////
        // Backward Propagation End ///////////
        ////////////////////////////////////////////////////////////////
        
        ////////////////////////////////////////////////////////////////
        // Stochastic Gradient Descent //////////////
        ////////////////////////////////////////////////////////////////
        
        // Calculate average of error 
        for (int B = 0; B<num_batch; B++) {
            for(int i =0;i<layer_num-1;i++){
                for(int j = 0;j<node_num[i+1];j++){
                    for(int b=B*BATCH_SIZE; b<BATCH_SIZE*(B+1) && b<NUM_TRAIN_SIZE; b++) {
                        d_err_average[B][i][j]+=d_err[b][i][j];
                        f_w_x_average_test[B][i+1][j]+=f_w_x_test[b][i+1][j];
                    }

                    d_err_average[B][i][j]/=BATCH_SIZE;
                    f_w_x_average_test[B][i+1][j]/=BATCH_SIZE;
                }
            }
        }

        for (int B = 0; B<num_batch; B++) {
            for(int i = 0;i<x_dim;i++) {
                for(int b=B*BATCH_SIZE; b<BATCH_SIZE*(B+1) && b<NUM_TRAIN_SIZE; b++) {
                    f_w_x_average_test[B][0][i]+=f_w_x_test[b][0][i];
                }
                 f_w_x_average_test[B][0][i]/=BATCH_SIZE;
            }

        }
        // calculate initial X average 
        for(int B = 0;B<num_batch;B++){
            for(int i =0;i<layer_num-1;i++){
                for(int j = 0;j<node_num[i+1];j++){
                    for(int k=0;k<node_num[i];k++){
                        // printf("%f, %f\n", d_err_average[B][i][j], f_w_x_average_test[B][i][k]);
                        w[i][j][k]-=alpha*d_err_average[B][i][j]*f_w_x_average_test[B][i][k]/NUM_TRAIN_SIZE;
                        bias[i]-=alpha*d_err_average[B][i][j]/NUM_TRAIN_SIZE;
                    }
                }
            }
        }
        
        ////////////////////////////////////////////////////////////////
        // Stochastic Gradient Descent End //////////////
        ///////////////////////////////////////////////////////////////
        
        printf("Epoch %d, MSE: %f\n", epoch, mean_square_error);

        if(epoch==NUM_EPOCHS-1){
            for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){
                for (int i=0;i<x_dim;i++){
                    
                    printf("x[%d]: %f ", i, x[train_index][i]);
                    if(i==x_dim-1){
                        printf(":::\n");
                    }
                }

                for(int i = 0;i<node_num[layer_num-1];i++){
                    printf( "train_index: %d, f - y: %f, y: %f, f: %f\n",train_index,f_w_x_test[train_index][layer_num-1][i] - y[train_index],
                    y[train_index],
                    f_w_x_test[train_index][layer_num-1][i]);
                }
            }
        }

    }

    // for(int train_index = 0;train_index<NUM_TRAIN_SIZE;train_index++){

    double *** test_k = create_test_data_set(a, NUM_TEST_SIZE,x_dim);
    double ** test_x = test_k[0];
    double * test_y = test_k[1][0];
    double *** test_f_w_x = create_func_output_arr(NUM_TEST_SIZE,layer_num,node_num);
    double *** f_w_x_test_2 = create_func_output_arr_test(NUM_TEST_SIZE,layer_num,test_x,node_num);
    double test_mean_square_error=0;

    /// Put all test values into a neural Network
    /// where all weights are adjusted by the previous backward propagations.
    for(int test_index = 0;test_index<NUM_TEST_SIZE;test_index++){
        for(int j = 1;j<layer_num+1;j++){
            for(int k =0;k<node_num[j];k++){
                double * x_dot_w = calculate_dot(w[j-1][k], f_w_x_test_2[test_index][j-1],node_num[j-1]);
                f_w_x_test_2[test_index][j][k] = calculate_linear_arr(x_dot_w,bias[j],node_num[j]);
                free(x_dot_w);
            }
        }

        test_mean_square_error+=pow(test_y[test_index] - f_w_x_test_2[test_index][layer_num-2][0],2)/(2*NUM_TEST_SIZE);
                
    }

    // Print the diff between test_f_w_x (predict) & y (output).
    for(int test_index = 0;test_index<NUM_TEST_SIZE;test_index++){

        for (int i=0;i<x_dim;i++){
            
            printf("x[%d]: %f ", i, f_w_x_test_2[test_index][0][i]);
            if(i==x_dim-1){
                printf(":::\n");
            }
        }

        for(int i = 0;i<node_num[layer_num-1];i++){
            printf( "x: %f, test_index: %d, f - y: %f, y: %f, f: %f\n",f_w_x_test_2[test_index][0][0],test_index,f_w_x_test_2[test_index][layer_num-1][i] - test_y[test_index],
            test_y[test_index],
            f_w_x_test_2[test_index][layer_num-1][i]);
        }
    }
    
    for(int i=0;i<layer_num-1;i++){
        for (int j=0;j<node_num[i+1];j++){
            for(int k=0;k<node_num[i];k++){
                printf("i: %d, j: %d, k: %d, w: %f\n", i,j,k,w[i][j][k]);
            }
        }
    }

}
