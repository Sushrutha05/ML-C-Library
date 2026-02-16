#ifndef LOGISTIC_REG_H
#define LOGISTIC_REG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Logistic Regression model structure.
 *
 * Stores learned weights, bias, and training state.
 */
typedef struct
{
    size_t num_features;        /**< Number of input features */
    double *weights;            /**< Weight vector of size num_features */
    double bias;                /**< Bias term */
    size_t stopping_iteration;  /**< Iteration where training stopped */
    int trained;                /**< Flag indicating if model is trained */
} RegressionModel;

/**
 * @brief Configuration parameters for logistic regression training.
 */
typedef struct
{
    double learning_rate;           /**< Gradient descent step size */
    size_t num_iterations;          /**< Maximum number of iterations */
    double early_stopping_threshold;/**< Relative loss improvement threshold */
} RegressionConfig;

/**
 * @brief Creates a logistic regression model.
 *
 * @param num_features Number of input features.
 * @return Pointer to allocated RegressionModel, or NULL on failure.
 */
RegressionModel* logreg_create(size_t num_features);

/**
 * @brief Frees memory associated with the model.
 *
 * @param model Pointer to model.
 */
void logreg_free(RegressionModel *model);

/**
 * @brief Trains the logistic regression model using batch gradient descent.
 *
 * @param model Pointer to model.
 * @param X Flattened feature matrix (size: num_samples * num_features).
 * @param y Target vector (size: num_samples), values must be 0 or 1.
 * @param num_samples Number of training samples.
 * @param config Training configuration parameters.
 *
 * @return 0 on success, -1 on failure.
 */
int logreg_train(RegressionModel *model,
                 const double *X,
                 const double *y,
                 size_t num_samples,
                 const RegressionConfig *config);

/**
 * @brief Predicts probability for a single sample.
 *
 * @param model Trained model.
 * @param x Feature vector (size: num_features).
 * @return Predicted probability in range [0,1], or NAN if model not trained.
 */
double logreg_predict(const RegressionModel *model,
                            const double *x);

#ifdef __cplusplus
}
#endif

#endif // LOGISTIC_REG_H
