/*
This is the test code for shared 
*/
# include "../shared/average_func.h"
# include "../shared/min_max.h"
# include <assert.h>

void test_min();
void test_max();
void test_average();
void test_variance();

int main(void){

// Test find MAX function
    
    test_max();
    test_min();
    test_average();
    test_variance();

    return 0;
}

void test_max(){
    
    double  test[5] = {1,3,5,10,3};
    assert(findMax(test,5)==10);

}

void test_min(){

    double test[5] = {1,3,5,10,3};
    assert(findMin(test,5)==1);

}

void test_average(){

    double test[5] = {1,1,1,1,1};
    assert(calculate_average(test,5)==1);

}

void test_variance(){
    double test[5] = {1,1,1,1,1};
    double test_avg = calculate_average(test, 5);
    
    assert(calculate_uncorrelated_std(test,test_avg,5)==0);
}

