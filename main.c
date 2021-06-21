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


# include "lib/average_func.h"
# include "lib/generate_random_arr.h"
# include "lib/simple_neural_network.h"
# include "lib/quadratic_neural_network.h"

// TODO: Create a library that allows
// do the regression by inputting some data



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
    
    char user_input;
    do{
        printf("Linear regression[1]\n");
        printf("Simple neural network[2]\n");
        printf("Quadratic neural network[3]\n");
        printf("Exit [4]\n");
        printf("Which algorithm would you like to try?: \n");

        scanf("%c",&user_input);

        switch (user_input){
            case '1':
            case '2':
            case '3':
                // Go to 
                goto out_loop;
            case '4':
                printf("Now exiting the program.\n");
                return 0;
            default:
                printf("Please select your action.\n\n");
        }

    }while(1);

    // Our loop & dconduct a program
    out_loop:


    if(user_input=='1'){
        printf("This is the test program for simple linear regression\n.");
        // Genearete random array data.
        double ** r=generate_random_data(MIN_a_RANGE,
        MAX_a_RANGE,MIN_b_RANGE,
        MAX_b_RANGE,RAND_ERR_MIN,
        RAND_ERR_MAX,MAX_ARR_SIZE);

        double * x = r[0];
        double * y = r[1];

        // Getting a and b value;
        double a=r[2][0];
        double b=r[2][1];

        printf("y=%fx+%f\n",a,b);

        double avg_x=calculate_average(x,MAX_ARR_SIZE);
        double avg_y=calculate_average(y,MAX_ARR_SIZE);

        printf("avg_x: %f\n", avg_x);
        printf("avg_y: %f\n", avg_y);

        double u_std_x=calculate_uncorrelated_std(x,avg_x,MAX_ARR_SIZE);
        double u_std_y=calculate_uncorrelated_std(y,avg_y,MAX_ARR_SIZE);

        printf("uncorrelated standard deviation x: %f\n", u_std_x);
        printf("uncorrelated standard deviation y: %f\n", u_std_y);

        double s_cor_xy=calculate_sample_correlation_coefficient(x,y,
        avg_x,avg_y,
        u_std_x,u_std_y,MAX_ARR_SIZE
        );

        double beta = s_cor_xy*u_std_y/u_std_x;
        double alpha = avg_y - (avg_x*beta);
        
        printf("alpha: %f\n",alpha);
        printf("beta: %f\n", beta);

        printf("sample correlation coefficient y: %f\n", s_cor_xy);

        // Free memory once all process has finished.
        for(int i=0;i<NUM_VALUES;i++){
            free(r[i]);
        }

        free(r);
    }
    
    if(user_input=='2'){
        int neural_arr_size=5;
        // Now creates data for simple neural network model: 
        // Input ==> Node ==> Output
        double ** arr_xy=simple_create_sample_input(neural_arr_size);
        
        // double * arr_weight=simple_create_random_weight(neural_arr_size);
        double w = -0.5;

        double error_rate = simple_calculate_error(w, arr_xy[1],neural_arr_size);
        
        printf("Error rate: %f\n", error_rate);

        double  d_err = simple_gradient_descent(w,arr_xy[0],arr_xy[1],neural_arr_size);
        double alpha=0.000001;

        int epoch_limit =100;
        for(int epoch=0;epoch<epoch_limit;epoch++){

            // Calculate error for one weight (forward propagation).
            double d_err = simple_gradient_descent(w,arr_xy[0],arr_xy[1],neural_arr_size);

            // Update weight (back propagation).
            w-=alpha*(d_err);
            if(epoch%10==0){
                printf("Epoch: %d\n", epoch);
                printf("Recalculated error rate: %f\n",simple_calculate_error(w, arr_xy[1],neural_arr_size));
                printf("d_err: %f\n", d_err);
                printf("Calculated weight value: %f\n", w);
            }
            
        }
        
        // The predicted function should be y = x/10+-0.1...

        free(arr_xy[0]);
        free(arr_xy[1]);
        free(arr_xy);

    }
    
    if(user_input='3'){
        int arr_length = 300;

        double *** a = quadratic_create_sample_input(arr_length);
        double ** x = a[0];
        double * y = a[1][0];
        double alpha = 0.00005;
        double * w = quadratic_create_random_sample_weight();

        // Set learning rate
        double learning_rate = 0.0001;
        int epoch_limit = 500;
        printf("Initial Phase");
        printf("w[0]: %f\n",w[0]);
        printf("w[1]: %f\n",w[1]);
        printf("w[2]: %f\n",w[2]);
        printf("==================\n");
        for (int epoch=0;epoch<epoch_limit;epoch++){
            
            double * d_err_w=quadratic_calculate_error(w,x,y,arr_length);
            
            // Update weight here.
            for(int i=0;i<INPUT_DIM;i++){
                w[i]+=alpha*(d_err_w[i]);
            }

            if(epoch%10 == 0){
                printf("epoch: %d\n", epoch);
                printf("w[0]: %f\n",w[0]);
                printf("w[1]: %f\n",w[1]);
                printf("w[2]: %f\n",w[2]);
            }
        }

    }


}

