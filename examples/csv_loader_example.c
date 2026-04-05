#include "csv_loader.h"
#include "mllib.h"
#include <stdio.h>

int main()
{
    Dataset data = load_csv("D:\\Sushrutha\\Projects\\ML model in C\\examples\\so.csv");

    if (data.rows == 0) {
        printf("Failed to load dataset\n");
        return 1;
    }

    MLModel *model = ml_create(ML_LINEAR, data.features);

    if (!model) {
        printf("Model creation failed\n");
        return 1;
    }

    RegressionConfig config = {
        .learning_rate = 0.01,
        .num_iterations = 1000,
    };

    ml_train(model, data.X, data.y, data.rows, &config);

    // Test prediction
    double sample1[] = {2, 3};
    double sample2[] = {10, 5};
    double pred1 = ml_predict(model, sample1);
    double pred2 = ml_predict(model, sample2);

    printf("Prediction for {2,3}: %.4f\n", pred1);
    printf("Prediction for {10,5}: %.4f\n", pred2);

    ml_destroy(model);
    free_dataset(&data);

    return 0;
}