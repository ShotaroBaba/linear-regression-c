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
void main() {
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

    printf("sample correlation coefficient y: %f", s_cor_xy);

    // Free memory once all process has finished.
    for(int i=0;i<NUM_VALUES;i++){
        free(r[i]);
    }

    free(r);


    // Now creates data for simple neural network model: 
    // Input ==> Node ==> Output
    double** arr_xy=create_sample_input(100);
    double * arr_weight=create_random_weight(100);

    free(arr_xy[0]);
    free(arr_xy[1]);
    free(arr_xy);
    
}

