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
#define NUM_TRAIN_SIZE 5000
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
    
    double alpha = 0.1;
    double *** w = quadratic_create_random_sample_weight();
    double *** initial_input = quadratic_create_sample_input(NUM_TRAIN_SIZE,10);

}
