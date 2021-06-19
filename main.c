# include <stdio.h>
# include <stdlib.h>
# include <time.h>

// Define default values
#define MAX_ARR_SIZE 500
#define RAND_ERR_MIN -0.5
#define RAND_ERR_MAX 0.5
#define NUM_VALUES 3
#define MIN_a_RANGE 1
#define MIN_b_RANGE 1
#define MAX_a_RANGE 10
#define MAX_b_RANGE 10
//////////////////////////



double ** generate_random_data();
double rand_double(double a, double b);
double * create_array();


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

    for(int i=0;i<MAX_ARR_SIZE;i++){
        printf("%d: (%f,%f)\n", i+1,x[i],y[i]);
    }

    // Free all values
    for(int i=0;i<NUM_VALUES;i++){
        free(r[i]);
    }

    free(r);

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
        y_rand=a*x_rand+b+rand_double(0,RAND_ERR_MAX);
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