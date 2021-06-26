# ifndef UTIL
# define UTIL
# include <math.h>

// find maximum value on the array
double find_max_value(double * arr, int size) {

    int i;
    int maxValue = arr[0];

    for (i = 1; i < size; ++i) {
        if ( arr[i] > maxValue ) {
            maxValue = arr[i];
        }
    }
    return maxValue;
}

double calculate_sigmoid(double a){
    return 1/(1+exp(-a));

}

double calculate_diff_sigmoid(double a){
    return calculate_sigmoid(a)*(1-calculate_sigmoid(a));
}


# endif