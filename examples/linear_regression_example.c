#include <stdio.h>
#include "mllib.h"

int main() {
    double x_vals[] = {1, 2, 3, 4, 5};
    double y_vals[] = {0.5, 0, -0.5, -1.0, -1.5};
    
    int num_elements = 5;
    double predict_x = 6;

    // "double" assumes the implementation handles string comparison for type
    // Ensure lin_reg signature matches headers
    printf("Predicted value for x=%.2f: %.2f\n", predict_x, lin_reg(x_vals, y_vals, num_elements, predict_x, "double"));

    return 0;
}
