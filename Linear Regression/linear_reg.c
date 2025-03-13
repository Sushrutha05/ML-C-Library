#include <stdio.h>
#include <string.h>  
#include <math.h>

double lin_reg(const void* x_arr, const void* y_arr, int no_ele, double x, const char* type) {
    if (no_ele < 2) {
        fprintf(stderr, "Error: Need at least 2 data points for regression.\n");
        return NAN;
    }

    double sum_x = 0, sum_y = 0, sum_XY = 0, sum_X2 = 0;

   
    for (int i = 0; i < no_ele; i++) {
        double x_val = (strcmp(type, "double") == 0) ? ((double*)x_arr)[i] : ((int*)x_arr)[i];
        double y_val = (strcmp(type, "double") == 0) ? ((double*)y_arr)[i] : ((int*)y_arr)[i];

        sum_x += x_val;
        sum_y += y_val;
    }

    double mean_x = sum_x / no_ele;
    double mean_y = sum_y / no_ele;

    
    for (int i = 0; i < no_ele; i++) {
        double x_val = (strcmp(type, "double") == 0) ? ((double*)x_arr)[i] : ((int*)x_arr)[i];
        double y_val = (strcmp(type, "double") == 0) ? ((double*)y_arr)[i] : ((int*)y_arr)[i];

        double x_diff = x_val - mean_x;
        double y_diff = y_val - mean_y;
        
        sum_XY += x_diff * y_diff;
        sum_X2 += x_diff * x_diff;
    }

    if (sum_X2 == 0) {
        fprintf(stderr, "Error: All x values are the same, cannot perform regression.\n");
        return  NAN;
    }

    double b_YX = sum_XY / sum_X2;
    double y = (b_YX * (x - mean_x)) + mean_y;

    return y;
}
