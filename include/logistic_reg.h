#ifndef LOGISTIC_REG_H
#define LOGISTIC_REG_H

#ifdef __cplusplus
extern "C" {
    #endif

/**
 * Returns the probability
 * 
 * @param z value to find the probability of
*/
double sigmoid(double z);

/**
 * Returns the loss
 * 
 * @param p value
 * @param y value
*/
double binary_cross_entropy(double p, double y);

/**
 * Initializ the model values 
 * 
 * @param num features which denote the number of features
*/
LogRegModel* logreg_create(size_t num_features);


/**
 * Deletes / frees the model 
 * 
 * @param num features which denote the number of features
*/
void logreg_free(LogRegModel *model);

/**
 * Trains the model to compute the weights and biases 
 * 
 * @param LogRegModel model which will be trained
 * @param Array An array of data points X
 * @param Array An array of data points Y
 * @param Int num_samples Number of samples
 * @param LogRegConfig config file for that model 
*/
int logreg_train(LogRegModel *model, const double *X, const double *y, size_t num_samples, const LogRegConfig config);

/**
 * Predicts probabilities based on the trained model
 * 
 * @param LogRegModel model which is trained
 * @param Double An elements whose image probability has to be predicted
*/
double logreg_predict_proba(const LogRegModel *model, const double *x);

/**
 * Predicts the class of the pre-image  based on trained model
 * 
 * @param LogRegModel model which is trained
 * @param Double An elements whose image probability has to be predicted
 * @param Double A threshold to determine to which class the image belongs to
*/
int logreg_predict(const LogRegModel *model, const double *x, double threshold);

#ifdef __cplusplus
}
#endif

#endif // LOGISTIC_REG_H