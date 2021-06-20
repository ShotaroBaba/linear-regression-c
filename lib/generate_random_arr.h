# ifndef GENERATE_RANDOM_ARR
# define GENERATE_RANDOM_ARR
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

double rand_double(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double r = random * (b - a);
    return a + r;
}

double * create_array(int maxArrSize) {
    double * values= (double *)calloc(maxArrSize, sizeof(double));
    return values;
}

// Generate random data for linear regression
double ** generate_random_data(
    int minARange,int maxARange, 
    int minBRange,int maxBRange,
    double randErrMin,double randErrMax,
    double maxArrSize
){

    double x_rand;
    double y_rand;
    
    // A valriable that stores all pointers.
    double ** k= (double **)calloc(sizeof(double*),3);

    double * x = create_array(maxArrSize);
    double * y = create_array(maxArrSize);
    double * ab = (double *)calloc(sizeof(double),2);

    srand( (unsigned)time(NULL));
    // Create random linear function y = ax + b
    int a = rand()% (maxARange-minARange+1)+minARange;
    int b = rand()% (minBRange-maxBRange+1)+minBRange;

    for(int i=0;i<maxArrSize;i++){
        x_rand=rand_double(0,8);
        y_rand=a*x_rand+b+rand_double(randErrMin,randErrMax);
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

# endif