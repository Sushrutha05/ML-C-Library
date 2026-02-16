#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
typedef struct
{
    size_t num_features;
    double *weights;
    double bias;
    size_t stopping_iteration;
    int trained;
} LinRegModel;
typedef struct
{
    double learning_rate;
    size_t num_iterations;
    double early_stopping_threshold;
} LinRegConfig;

LinRegModel* linreg_create(size_t num_features)
{
    LinRegModel *model = malloc(sizeof(LinRegModel));
    if(!model) return NULL;

    model->weights = calloc(num_features, sizeof(double));
    if(!model->weights){
        free(model);
        return NULL;
    }

    model->num_features = num_features;
    model->bias = 0.0;
    model->trained = 0;
    model->stopping_iteration = 0;
    return model;
}

void linreg_free(LinRegModel *model)
{
    if (!model)
        return;

    free(model->weights);
    free(model);
}

void linreg_train(LinRegModel *model, const double *x, const double *y, const size_t num_samples, const LinRegConfig config)
{
    if (model == NULL || x == NULL || y == NULL)
    {
        fprintf(stderr, "Null pointer passed to linreg_train.\n");
        return;
    }

    if (num_samples < 2)
    {
        fprintf(stderr, "Error: Need at least 2 data points for regression.\n");
        return;
    }

    if (!model->weights)
    {
        fprintf(stderr, "Model weights not initialized properly.\n");
        return;
    }
    model->trained = 0;

    double *dw = calloc(model->num_features, sizeof(double));

    if (!dw)
    {
        fprintf(stderr, "Memory Allocation failed.\n");
        return;
    }

    double prev_loss = DBL_MAX;

    model->stopping_iteration = config.num_iterations;

    for (size_t iter = 0; iter < config.num_iterations; iter++)
    {

        memset(dw, 0, model->num_features * sizeof(double));
        double db = 0.0;
        double curr_loss = 0.0;
        
        
        for (size_t i = 0; i < num_samples; i++)
        {
            double y_pred = model->bias;

            for (size_t j = 0; j < model->num_features; j++)
            {
                y_pred += model->weights[j] * x[i * model->num_features + j];
            }

            double error = y_pred - y[i];

            for (size_t j = 0; j < model->num_features; j++)
            {
                dw[j] += error * x[i * model->num_features + j];
            }

            db += error;
            curr_loss += error * error;
        }
        curr_loss /= (2 * num_samples);
        if (prev_loss > 0 && fabs(prev_loss - curr_loss) / prev_loss < config.early_stopping_threshold)
        {
            model->stopping_iteration = iter;
            break;
        }

        for (size_t j = 0; j < model->num_features; j++)
        {
            model->weights[j] -= config.learning_rate * (dw[j] / num_samples);
        }

        model->bias -= config.learning_rate * (db / num_samples);

        prev_loss = curr_loss;
    }
    free(dw);
    model->trained = 1;
}

double linreg_predict(LinRegModel *model, const double *x)
{

    if (model == NULL || !model->trained)
    {
        fprintf(stderr, "Model not trained.\n");
        return NAN;
    }

    if (!x)
    {
        fprintf(stderr, "Null feature vector.\n");
        return NAN;
    }

    double y_pred = model->bias;

    for (size_t j = 0; j < model->num_features; j++)
    {
        y_pred += model->weights[j] * x[j];
    }

    return y_pred;
}
