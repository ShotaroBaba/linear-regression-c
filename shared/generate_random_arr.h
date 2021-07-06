# ifndef GENERATE_RANDOM_ARR
# define GENERATE_RANDOM_ARR
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

double rand_double(double a, double b);

double rand_double(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double r = random * (b - a);
    return a + r;
}

double * create_array(int maxArrSize) {
    double * values= (double *)calloc(maxArrSize, sizeof(double));
    return values;
}


# endif /* GENERATE_RANDOM_ARR */