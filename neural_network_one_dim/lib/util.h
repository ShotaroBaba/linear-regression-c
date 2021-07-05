# ifndef UTIL
# define UTIL
# include <math.h>


double calculate_sigmoid(double a){
    return 1/(1+exp(-a));

}

double calculate_diff_sigmoid(double a){
    return calculate_sigmoid(a)*(1-calculate_sigmoid(a));
}

double calculate_sigmoid_arr(double *a,double bias, int arrSize) {

    double tmp_sum = 0;
    for(int i = 0;i<arrSize;i++){
        tmp_sum+=a[i];
    }
    tmp_sum+=bias;
    return 1/(1+exp(-tmp_sum));
}

double calculate_sigmoid_diff_arr(double *a,double bias,int arrSize){
    return calculate_sigmoid_arr(a,bias,arrSize)*(1-calculate_sigmoid_arr(a,bias,arrSize));
}

double calculate_relu_arr(double *a, int arrSize){
    
    double tmp_sum = 0;
    for(int i = 0;i<arrSize;i++){
        tmp_sum+=a[i];
    }

    return tmp_sum;
}

double calculate_linear_arr(double *a,double bias, int arrSize){
    double tmp_sum = 0;
    for(int i = 0;i<arrSize;i++){
        tmp_sum += a[i];
    }
    tmp_sum+=bias;
    return tmp_sum;
}

double * calculate_dot(double *a, double *b, int arrSize){

    // Check if both size is the same...
    int a_size=sizeof(a)/sizeof(double);
    int b_size=sizeof(b)/sizeof(double);

    if(a_size!=b_size){
        printf("Error: The sizes of two arrays are not the same.\n");
        return NULL;
    }

    double * dot_result= calloc(arrSize, sizeof(double));
    

    for(int i =0; i<arrSize; i++){
        dot_result[i]=a[i]*b[i];
    }

    return dot_result;
}

# endif