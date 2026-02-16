#ifndef LINEAR_REG_H
#define LINEAR_REG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear Regression model structure.
 *
 * Stores learned weights, bias, and training state.
 */
typedef struct
{
    size_t num_features;      /**< Number of input features */
    double *weights;          /**< Weight vector of size num_features */
    double bias;              /**< Bias term */
    size_t stopping_iteration;/**< Iteration where training stopped */
    int trained;              /**< Flag indicating if model is trained */
} LinRegModel;

/**
 * @brief Configuration parameters for training.
 *
 * Controls learning rate, number of iterations,
 * and early stopping behavior.
 */
typedef struct
{
    double learning_rate;          /**< Gradient descent step size */
    size_t num_iterations;         /**< Maximum number of iterations */
    double early_stopping_threshold; /**< Relative loss improvement threshold */
} LinRegConfig;

/**
 * @brief Creates a linear regression model.
 *
 * @param num_features Number of input features.
 * @return Pointer to allocated LinRegModel or NULL on failure.
 */
LinRegModel* linreg_create(size_t num_features);

/**
 * @brief Trains the model using batch gradient descent.
 *
 * @param model Pointer to model.
 * @param X Flattened feature matrix (size: num_samples * num_features).
 * @param y Target vector (size: num_samples).
 * @param num_samples Number of training samples.
 * @param config Training configuration.
 */
void linreg_train(LinRegModel *model,
                  const double *X,
                  const double *y,
                  size_t num_samples,
                  LinRegConfig config);

/**
 * @brief Predicts output for a single sample.
 *
 * @param model Trained model.
 * @param x Feature vector (size: num_features).
 * @return Predicted value or NAN if model not trained.
 */
double linreg_predict(LinRegModel *model,
                      const double *x);

/**
 * @brief Frees model memory.
 *
 * @param model Pointer to model.
 */
void linreg_free(LinRegModel *model);

#ifdef __cplusplus
}
#endif

#endif
