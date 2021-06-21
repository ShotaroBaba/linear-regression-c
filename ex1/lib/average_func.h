# ifndef AVERAGE_FUNC
# define AVERAGE_FUNC

double calculate_uncorrelated_std(double *a, double a_avg, int arrSize){

    double total=0;
    
    for(int i=0;i<arrSize;i++){
        total+=pow((a[i] - a_avg),2)/arrSize;
        // printf("Calc progress: %f\n",total);
        // printf("a[%d]: %f, a[%d]-avg: %f\n", i, a[i],i, (a[i]-a_avg)/(double)MAX_ARR_SIZE);
    }
    
    total=sqrt(total);

    return total;
}

double calculate_sample_correlation_coefficient(double *a, double *b,
double a_avg, double b_avg,
double u_std_a, double u_std_b,
int arrSize){

    double total=0;
    
    for (int i=0;i<arrSize;i++) {
        total+=( (a[i]-a_avg)*(b[i]-b_avg) )/((arrSize-1)*u_std_a*u_std_b);
    }

    return total;
}

double calculate_average(double *a,int arrSize) {
    
    double total=0;

    for(int i=0;i<arrSize;i++){
        total+=a[i]/(double)arrSize;
    }

    return total;
}

#endif /* AVERAGE_FUNC */