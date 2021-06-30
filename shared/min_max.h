# ifndef MIN_MAX
# define MIN_MAX

double findMax(double * arr, int arrSize);
double findMin(double * arr, int arrSize);


double findMax(double * arr, int arrSize)
{
    // Initialize maximum element
    double max = arr[0];

    if(arrSize==1){
        return max;
    }

    for (int i = 1; i < arrSize; i++){
        if (arr[i] > max) {
            max = arr[i];
        }
    
    }

    return max;
}

double findMin(double * arr, int arrSize){
     
    // Initialize maximum element
    double min = arr[0];

    if(arrSize==1){
        return min;
    }


    for (int i = 1; i < arrSize; i++){
        if (arr[i] < min){
            min = arr[i];
        }
            
    }

    return min;

}


# endif