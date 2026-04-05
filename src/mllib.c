#include "mllib.h"
#include <stdlib.h>

/*
 * Internal structure definition
 */
struct MLModel {
    MLModelType type;
    size_t num_features;
    void *model;  // points to RegressionModel
};

/*
 * Create model
 */
MLModel *ml_create(MLModelType type, size_t num_features)
{
    MLModel *m = malloc(sizeof(MLModel));
    if (!m) return NULL;

    m->type = type;
    m->num_features = num_features;
    m->model = NULL;

    switch (type)
    {
        case ML_LINEAR:
            m->model = linreg_create(num_features);
            break;

        case ML_LOGISTIC:
            m->model = logreg_create(num_features);
            break;

        default:
            free(m);
            return NULL;
    }

    if (!m->model) {
        free(m);
        return NULL;
    }

    return m;
}

/*
 * Train model
 */
int ml_train(MLModel *model,
             const double *x,
             const double *y,
             size_t num_samples,
             const RegressionConfig *config)
{
    if (!model || !model->model) return -1;

    switch (model->type)
    {
        case ML_LINEAR:
            return linreg_train((RegressionModel *)model->model,
                                x, y, num_samples, config);

        case ML_LOGISTIC:
            return logreg_train((RegressionModel *)model->model,
                                x, y, num_samples, config);

        default:
            return -1;
    }
}

/*
 * Predict
 */
double ml_predict(const MLModel *model,
                  const double *x)
{
    if (!model || !model->model) return 0.0;

    switch (model->type)
    {
        case ML_LINEAR:
            return linreg_predict((RegressionModel *)model->model, x);

        case ML_LOGISTIC:
            return logreg_predict((RegressionModel *)model->model, x);

        default:
            return 0.0;
    }
}

/*
 * Destroy model
 */
void ml_destroy(MLModel *model)
{
    if (!model) return;

    if (model->model)
    {
        switch (model->type)
        {
            case ML_LINEAR:
                linreg_free((RegressionModel *)model->model);
                break;

            case ML_LOGISTIC:
                logreg_free((RegressionModel *)model->model);
                break;
        }
    }

    free(model);
}