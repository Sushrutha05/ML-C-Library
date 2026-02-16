#include <stdio.h>
#include <stdlib.h>
#include "logistic_reg.h"

int main() {
    // Simple dataset: 1 feature. 
    // x < 5 => class 0
    // x >= 5 => class 1
    double X[] = { 
        1.0, 2.0, 3.0, 4.0, 
        6.0, 7.0, 8.0, 9.0 
    };
    double y[] = { 
        0.0, 0.0, 0.0, 0.0, 
        1.0, 1.0, 1.0, 1.0 
    };
    
    size_t num_samples = 8;
    size_t num_features = 1;

    // Configuration
    RegressionConfig config = {
        .learning_rate = 0.1,
        .num_iterations = 5000,
        .early_stopping_threshold = 1e-6
    };

    printf("Creating Logistic Regression Model...\n");
    RegressionModel *model = logreg_create(num_features);
    if (!model) {
        fprintf(stderr, "Failed to create model.\n");
        return 1;
    }

    printf("Training...\n");
    if (logreg_train(model, X, y, num_samples, &config) != 0) {
        fprintf(stderr, "Training failed.\n");
        logreg_free(model);
        return 1;
    }

    printf("Model Trained.\n");
    printf("Bias: %f\n", model->bias);
    printf("Weight: %f\n", model->weights[0]);

    // Predictions
    double test_low[] = { 2.5 };
    double prob_low = logreg_predict(model, test_low);
    printf("Probability for x=2.5 (Expect ~0): %f\n", prob_low);

    double test_high[] = { 7.5 };
    double prob_high = logreg_predict(model, test_high);
    printf("Probability for x=7.5 (Expect ~1): %f\n", prob_high);

    logreg_free(model);
    return 0;
}
