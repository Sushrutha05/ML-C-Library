#include <stdio.h>
#include "mllib.h"

int main() {
    double X[] = {
        0, 1,
        1, 2,
        2, 3,
        3, 4
    };

    double y[] = {0, 0, 1, 1};

    size_t num_samples = 4;
    size_t num_features = 2;

    LogRegConfig config = {
        .learning_rate = 0.1,
        .num_iterations = 1000,
        .early_stopping_threshold = 0.0001,
        .classification_threshold = 0.5
    };

    LogRegModel *model = logreg_create(num_features);

    if (!model) {
        fprintf(stderr, "Failed to create model.\n");
        return -1;
    }

    if (logreg_train(model, X, y, num_samples, config) != 0) {
        fprintf(stderr, "Training failed.\n");
        logreg_free(model);
        return -1;
    }

    printf("Trained Weights:\n");
    for (size_t j = 0; j < model->num_features; j++) {
        printf("w%zu = %.6f\n", j, model->weights[j]);
    }

    printf("bias = %.6f\n", model->bias);

    double test_sample[] = {2, 3};

    double prob = logreg_predict_proba(model, test_sample);
    int cls = logreg_predict(model,
                             test_sample,
                             config.classification_threshold);

    printf("\nTest Sample [2, 3]:\n");
    printf("probability = %.6f\n", prob);
    printf("class = %d\n", cls);

    logreg_free(model);

    return 0;
}
