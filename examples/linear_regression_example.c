#include <stdio.h>
#include <stdlib.h>
#include "linear_reg.h"

int main() {
    // Data: y = 2x
    double X[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    double y[] = { 2.0, 4.0, 6.0, 8.0, 10.0 };
    
    size_t num_samples = 5;
    size_t num_features = 1;

    // Configuration
    RegressionConfig config = {
        .learning_rate = 0.01,
        .num_iterations = 2000,
        .early_stopping_threshold = 1e-6
    };

    printf("Creating Linear Regression Model...\n");
    RegressionModel *model = linreg_create(num_features);
    if (!model) {
        fprintf(stderr, "Failed to create model.\n");
        return 1;
    }

    printf("Training...\n");
    if (linreg_train(model, X, y, num_samples, &config) != 0) {
        fprintf(stderr, "Training failed.\n");
        linreg_free(model);
        return 1;
    }

    printf("Model Trained.\n");
    printf("Bias: %f\n", model->bias);
    printf("Weight: %f\n", model->weights[0]);

    // Prediction
    double test_val[] = { 6.0 };
    double prediction = linreg_predict(model, test_val);
    printf("Prediction for x=6.0: %f\n", prediction);

    linreg_free(model);
    return 0;
}
