#include <stdlib.h>
#include <math.h>
#include <float.h>

static const double LOGREG_EPSILON = 1e-15;

typedef struct {
    double *weights;
    size_t num_features;
    double bias;
    size_t stopping_iteration;
} LogRegModel;

typedef struct {
    double learning_rate;
    size_t num_iterations;
    double early_stopping_threshold;
    double classification_threshold;
} LogRegConfig;

double sigmoid(double z) {
    if (z >= 0) {
        double exp_neg = exp(-z);
        return 1.0 / (1.0 + exp_neg);
    } else {
        double exp_pos = exp(z);
        return exp_pos / (1.0 + exp_pos);
    }
}

double binary_cross_entropy(double p, double y) {
    p = fmax(LOGREG_EPSILON, fmin(1.0 - LOGREG_EPSILON, p));
    return -((y * log(p)) + ((1 - y) * log(1 - p)));
}

LogRegModel* logreg_create(size_t num_features) {
    LogRegModel *model = malloc(sizeof(LogRegModel));
    if (!model) return NULL;

    model->weights = calloc(num_features, sizeof(double));
    if (!model->weights) {
        free(model);
        return NULL;
    }

    model->num_features = num_features;
    model->bias = 0.0;
    model->stopping_iteration = 0;

    return model;
}

void logreg_free(LogRegModel *model) {
    if (!model) return;
    free(model->weights);
    free(model);
}

void logreg_train(LogRegModel *model,
                 const double *X,
                 const double *y,
                 size_t num_samples,
                 LogRegConfig config) {

    if (!model || !X || !y || num_samples == 0)
        return -1;

    double prev_loss = DBL_MAX;

    for (size_t iter = 0; iter < config.num_iterations; iter++) {

        double *dw = calloc(model->num_features, sizeof(double));
        if (!dw) return -1;

        double db = 0.0;

        for (size_t i = 0; i < num_samples; i++) {

            double z = model->bias;

            for (size_t j = 0; j < model->num_features; j++) {
                z += model->weights[j] *
                     X[i * model->num_features + j];
            }

            double p = sigmoid(z);
            double err = p - y[i];

            for (size_t j = 0; j < model->num_features; j++) {
                dw[j] += err *
                         X[i * model->num_features + j];
            }

            db += err;
        }

        for (size_t j = 0; j < model->num_features; j++) {
            dw[j] /= num_samples;
            model->weights[j] -= config.learning_rate * dw[j];
        }

        db /= num_samples;
        model->bias -= config.learning_rate * db;

        double total_loss = 0.0;

        for (size_t i = 0; i < num_samples; i++) {

            double z = model->bias;

            for (size_t j = 0; j < model->num_features; j++) {
                z += model->weights[j] *
                     X[i * model->num_features + j];
            }

            double p = sigmoid(z);
            total_loss += binary_cross_entropy(p, y[i]);
        }

        double avg_loss = total_loss / num_samples;

        free(dw);

        if (fabs(prev_loss - avg_loss) <
            config.early_stopping_threshold) {
            model->stopping_iteration = iter + 1;
            break;
        }

        prev_loss = avg_loss;
        model->stopping_iteration = iter + 1;
    }
}

double logreg_predict_proba(const LogRegModel *model,
                            const double *x) {

    double z = model->bias;

    for (size_t j = 0; j < model->num_features; j++) {
        z += model->weights[j] * x[j];
    }

    return sigmoid(z);
}

int logreg_predict(const LogRegModel *model,
                   const double *x,
                   double threshold) {

    double p = logreg_predict_proba(model, x);
    return (p >= threshold) ? 1 : 0;
}

