# ifndef MIN_MAX
# define MIN_MAX

double findMax(double * arr, int arrSize);
double findMin(double * arr, int arrSize);


double findMax(double * arr, int arrSize)
{
    int i;
     
    // Initialize maximum element
    int max = arr[0];

    for (i = 1; i < arrSize; i++)
        if (arr[i] > max) {
            max = arr[i];
        }
            max = arr[i];
    
    return max;
}

double findMin(double * arr, int arrSize){
    int i;
     
    // Initialize maximum element
    int min = arr[0];

    for (i = 1; i < arrSize; i++)
        if (arr[i] < min){
            min = arr[i];
        }
            
    return min;

}


# endif