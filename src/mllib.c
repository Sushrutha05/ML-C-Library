#include "linear_reg.h"
#include "logistic_reg.h"

#include <stdio.h>
#include <stdlib.h>

typedef enum
{
    ML_LINEAR,
    ML_LOGISTIC
} MLModelType;

typedef struct
{
    MLModelType type;
    void *internal_model_pointer;
    int (*train)(RegressionModel *, const double *, const double *, const size_t, const RegressionConfig *);
    double (*predict)(const RegressionModel*, const double *);
    void (*destroy)(RegressionModel *);
} MLModel;

MLModel *ml_create(MLModelType type, size_t num_features)
{
    MLModel *model = (MLModel *)malloc(sizeof(MLModel));
    if (!model)
    {
        fprintf(stderr, "ML Model creation failed\n");
        return NULL;
    }

    model->type = type;

    switch (type)
    {
    case ML_LINEAR:

        model->internal_model_pointer = linreg_create(num_features);
        model->train = linreg_train;
        model->predict = linreg_predict;
        model->destroy = linreg_free;
        break;
    case ML_LOGISTIC:

        model->internal_model_pointer = logreg_create(num_features);
        model->train = logreg_train;
        model->predict = logreg_predict;
        model->destroy = logreg_free;
        break;
    default:
        fprintf(stderr, "Invalid Model Type.\n");
        free(model);
        return NULL;
    }

    if (!model->internal_model_pointer)
    {
        free(model);
        return NULL;
    }
    return model;
}

int ml_train(MLModel *model, const double *x, const double *y, const size_t num_samples, const void *config)
{
    if (!model || !model->internal_model_pointer)
    {
        fprintf(stderr, "Invalid ML Model.\n");
        return -1;
    }

    return model->train(model->internal_model_pointer, x, y, num_samples, config);
}

double ml_predict(MLModel *model, const double *x)
{
    if (!model || !model->internal_model_pointer)
    {
        fprintf(stderr, "Invalid ML Model.\n");
        return NAN;
    }

    return model->predict(model->internal_model_pointer, x);
}

void ml_destroy(MLModel *model)
{
    if (!model)
    {
        fprintf(stderr, "Invalid Model.\n");
        return;
    }

    model->destroy(model->internal_model_pointer);
    free(model);
}