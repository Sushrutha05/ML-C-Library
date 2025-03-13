#ifndef LINEAR_REG_H
#define LINEAR_REG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Performs simple linear regression and predicts y for a given x.
 * 
 * @param x_arr Pointer to an array of x values (int* or double*).
 * @param y_arr Pointer to an array of y values (int* or double*).
 * @param no_ele Number of elements in the arrays.
 * @param x The x value for which to predict y.
 * @param type The data type of x_arr and y_arr ("int" or "double").
 */
double lin_reg(const void* x_arr, const void* y_arr, int no_ele, double x, const char* type);

#ifdef __cplusplus
}
#endif

#endif // LINEAR_REG_H