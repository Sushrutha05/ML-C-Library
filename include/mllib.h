#ifndef MLLIB_H
#define MLLIB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Forward declaration of RegressionConfig */
typedef struct RegressionConfig RegressionConfig;

/*
 * Supported model types
 */
typedef enum
{
    ML_LINEAR,
    ML_LOGISTIC
} MLModelType;

/*
 * Opaque ML model structure.
 * Internal details are hidden from the user.
 */
typedef struct MLModel MLModel;

/**
 * Creates a machine learning model.
 *
 * @param type          Type of model (ML_LINEAR or ML_LOGISTIC)
 * @param num_features  Number of input features
 *
 * @return Pointer to MLModel on success, NULL on failure
 */
MLModel *ml_create(MLModelType type, size_t num_features);

/**
 * Trains the given model.
 *
 * @param model         Pointer to MLModel
 * @param x             Flattened feature matrix (row-major)
 * @param y             Target values array
 * @param num_samples   Number of training samples
 * @param config        Pointer to RegressionConfig
 *
 * @return 0 on success, -1 on failure
 */
int ml_train(MLModel *model,
             const double *x,
             const double *y,
             size_t num_samples,
             const RegressionConfig *config);

/**
 * Predicts output using trained model.
 *
 * @param model   Pointer to MLModel
 * @param x       Feature vector for single sample
 *
 * @return Prediction value (linear output or logistic probability)
 */
double ml_predict(const MLModel *model,
                  const double *x);

/**
 * Destroys the model and frees memory.
 *
 * @param model   Pointer to MLModel
 */
void ml_destroy(MLModel *model);

#ifdef __cplusplus
}
#endif

#endif /* MLLIB_H */
