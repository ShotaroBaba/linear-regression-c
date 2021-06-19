# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <math.h>

// TODO: Create a library that allows
// do the regression by inputting some data.

// Define default values
#define MAX_ARR_SIZE 100000
#define RAND_ERR_MIN -33.5
#define RAND_ERR_MAX 33.5
#define NUM_VALUES 3
#define MIN_a_RANGE 1
#define MIN_b_RANGE 1
#define MAX_a_RANGE 10
#define MAX_b_RANGE 10
//////////////////////////


// Declaring functions ///////////
double ** generate_random_data();
double rand_double(double a, double b);
double * create_array();
double calculate_average(double *a);
double calculate_uncorrelated_std(double *a, double a_avg);
double calculate_sample_correlation_coefficient(double *a, double *b,
double avg_x, double avg_y,
double u_std_a, double u_std_b);
//////////////////////////////////

// Simple linear regression
void main() {
    printf("This is the test program for simple linear regression\n.");
    // Genearete random array data.
    double ** r=generate_random_data();

    double * x = r[0];
    double * y = r[1];

    // Getting a and b value;
    double a=r[2][0];
    double b=r[2][1];

    printf("y=%fx+%f",a,b);

    double avg_x=calculate_average(x);
    double avg_y=calculate_average(y);

    printf("avg_x: %f\n", avg_x);
    printf("avg_y: %f\n", avg_y);

    double u_std_x=calculate_uncorrelated_std(x,avg_x);
    double u_std_y=calculate_uncorrelated_std(y,avg_y);

    printf("uncorrelated standard deviation x: %f\n", u_std_x);
    printf("uncorrelated standard deviation y: %f\n", u_std_y);

    double s_cor_xy=calculate_sample_correlation_coefficient(x,y,
    avg_x,avg_y,
    u_std_x,u_std_y
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

}

double calculate_uncorrelated_std(double *a, double a_avg){

    double total=0;

    for(int i=0;i<MAX_ARR_SIZE;i++){
        total+=pow((a[i] - a_avg),2)/MAX_ARR_SIZE;
        // printf("Calc progress: %f\n",total);
        // printf("a[%d]: %f, a[%d]-avg: %f\n", i, a[i],i, (a[i]-a_avg)/(double)MAX_ARR_SIZE);
    }
    
    total=sqrt(total);

    return total;
}

double calculate_sample_correlation_coefficient(double *a, double *b,
double a_avg, double b_avg,
double u_std_a, double u_std_b){
    double total=0;

    for (int i=0;i<MAX_ARR_SIZE;i++) {
        total+=( (a[i]-a_avg)*(b[i]-b_avg) )/((MAX_ARR_SIZE-1)*u_std_a*u_std_b);
    }

    return total;
}

double calculate_average(double *a) {
    
    double total=0;
    for(int i=0;i<MAX_ARR_SIZE;i++){
        total+=a[i]/(double)MAX_ARR_SIZE;
    }

    return total;
}

double rand_double(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double r = random * (b - a);
    return a + r;
}

double * create_array() {
    double * values= calloc(MAX_ARR_SIZE, sizeof(double));

    return values;
}

// Generate random data for linear regression
double ** generate_random_data(){

    double x_rand;
    double y_rand;
    
    // A valriable that stores all pointers.
    double ** k= calloc(sizeof(double*),3);

    double * x = create_array();
    double * y = create_array();
    double * ab = calloc(sizeof(double),2);

    srand( (unsigned)time(NULL));
    // Create random linear function y = ax + b
    int a = rand()%(MAX_a_RANGE-MIN_a_RANGE+1)+MIN_a_RANGE;
    int b = rand()%(MAX_b_RANGE-MIN_b_RANGE+1)+MIN_b_RANGE;

    for(int i=0;i<MAX_ARR_SIZE;i++){
        x_rand=rand_double(0,8);
        y_rand=a*x_rand+b+rand_double(RAND_ERR_MIN,RAND_ERR_MAX);
        x[i]=x_rand;
        y[i]=y_rand;
    }

    ab[0]=a;
    ab[1]=b;

    k[0]=x;
    k[1]=y;
    k[2]=ab;

    return k;
}