# MLLib: Modular Machine Learning Library in C

**MLLib** is a lightweight, modular machine learning library written in C. It provides implementations for fundamental algorithms like Linear Regression and Logistic Regression, featuring a clean API, manual memory management, and a flexible training configuration.

## 📂 Project Structure

```
ML model in C/
├── include/                # Public header files
│   ├── linear_reg.h        # Linear Regression API
│   ├── logistic_reg.h      # Logistic Regression API
│   └── mllib.h             # Generic MLModel abstraction
├── src/                    # Source code implementations
│   ├── linear_reg.c        # Linear Regression implementation
│   ├── logistic_reg.c      # Logistic Regression implementation
│   └── mllib.c             # Generic MLModel wrapper
├── examples/               # Example usage
│   ├── linear_regression_example.c
│   └── logistic_regression_example.c
├── lib/                    # Compiled static library output
├── build.bat               # Windows build script
└── README.md               # Project documentation
```

## 🏗 Architecture

The library is designed with modularity and extensibility in mind.

### 1. Specific Model Implementations
Each algorithm is implemented in its own translation unit (e.g., `src/linear_reg.c`, `src/logistic_reg.c`) with a corresponding header. These modules are self-contained and share a similar "Entity-Component" style structure:
- **Model Struct**: Holds the state (weights, bias, training status).
- **Config Struct**: Parameterizes the training process (learning rate, iterations).
- **Functions**: `_create`, `_train`, `_predict`, `_free`.

### 2. Abstraction Layer (`MLModel`)
The file `include/mllib.h` and `src/mllib.c` define a generic `MLModel` structure that acts as a polymorphic wrapper around specific implementations. 

**Internal Working of MLModel:**
It uses a v-table (virtual table) approach using function pointers to achieve polymorphism in C.
- **`internal_model_pointer`**: A `void*` pointing to the actual model instance (e.g., `RegressionModel*` from linear or logistic modules).
- **Function Pointers**:
  - `train`: wrapper calling the specific training function.
  - `predict`: wrapper calling the specific prediction function.
  - `destroy`: wrapper calling the specific free function.

When `ml_create(ML_LINEAR, ...)` is called, it allocates the specific linear model, assigns it to `internal_model_pointer`, and points the function pointers to `linreg_train`, `linreg_predict`, etc.

### 3. Memory Model
The library assumes explicit memory management:
- **Creation**: `*_create` functions allocate memory for the model structure and its internal arrays (weights).
- **Destruction**: `*_free` functions must be called by the user to release standard memory.
- **Data Ownership**: The library does **not** copy or own the training data (`X`, `y`). The user must ensure these arrays remain valid during the training call.

### 4. Data Layout
- **Feature Matrix (X)**: Flattened 1D array representing a 2D matrix in **row-major order**.
  - Size: `num_samples * num_features`
  - Access: `X[i * num_features + j]` (sample `i`, feature `j`).
- **Target Vector (y)**: Simple 1D array of size `num_samples`.

---

## � API Documentation

### Common Structures
Both Linear and Logistic regression usage share similar configuration structures.

**`RegressionConfig`**
- `learning_rate` (`double`): Step size for gradient descent.
- `num_iterations` (`size_t`): Maximum training epochs.
- `early_stopping_threshold` (`double`): Stop if loss improvement is smaller than this relative threshold.

### Linear Regression (`include/linear_reg.h`)

| Function | Description |
|----------|-------------|
| `RegressionModel* linreg_create(size_t num_features)` | Allocates and initializes a new model. |
| `void linreg_free(RegressionModel *model)` | Frees the model and its weights. |
| `int linreg_train(RegressionModel *model, const double *X, const double *y, size_t n, const RegressionConfig *cfg)` | Trains the model using Batch Gradient Descent (MSE loss). |
| `double linreg_predict(RegressionModel *model, const double *x)` | Predicts a continuous value for a given feature vector. |

### Logistic Regression (`include/logistic_reg.h`)

| Function | Description |
|----------|-------------|
| `RegressionModel* logreg_create(size_t num_features)` | Allocates and initializes a new model. |
| `void logreg_free(RegressionModel *model)` | Frees the model and its weights. |
| `int logreg_train(RegressionModel *model, const double *X, const double *y, size_t n, const RegressionConfig *cfg)` | Trains the model using Batch Gradient Descent (Log Loss). |
| `double logreg_predict(const RegressionModel *model, const double *x)` | Predicts probability [0, 1] for a given feature vector. |

---

## ⚙️ Training Implementation

The training process uses **Batch Gradient Descent**:

1. **Initialization**: Weights are zero-initialized; Bias is zero-initialized.
2. **Forward Pass**:
   - Compute linear hypothesis: `z = w*x + b`.
   - (Logistic only) Apply Sigmoid activation: `1 / (1 + exp(-z))`.
3. **Loss Calculation**:
   - **Linear**: Mean Squared Error (MSE).
   - **Logistic**: Binary Cross Entropy (Log Loss).
4. **Backward Pass (Gradient Calculation)**:
   - Compute gradients for weights (`dw`) and bias (`db`) averaged over all samples.
5. **Update**:
   - `w = w - learning_rate * dw`
   - `b = b - learning_rate * db`
6. **Early Stopping**:
   - Checks if `(prev_loss - curr_loss) / prev_loss < threshold`.
   - If converged, stops early to save computation.

---

## 🛠 Build Instructions

The project uses a simple batch script for building on Windows with MinGW/GCC.

### Prerequisites
- GCC Compiler (MinGW)
- Windows OS

### Building the Static Library
Run the `build.bat` script from the root directory:

```powershell
.\build.bat
```

This will:
1. Compile `src/linear_reg.c` and `src/logistic_reg.c` into object files.
2. Archive them into a static library `lib/libmllib.a`.
3. Compile the example programs in `examples/`.

---

## 💻 Usage Examples

### Linear Regression
```c
#include <stdio.h>
#include "linear_reg.h"

int main() {
    // Data: y = 2x
    double X[] = { 1, 2, 3, 4, 5 };
    double y[] = { 2, 4, 6, 8, 10 };
    
    RegressionConfig config = { .learning_rate = 0.01, .num_iterations = 2000, .early_stopping_threshold = 1e-6 };
    
    // 1. Create
    RegressionModel *model = linreg_create(1);
    
    // 2. Train
    linreg_train(model, X, y, 5, &config);
    
    // 3. Predict
    double test[] = { 6.0 };
    double pred = linreg_predict(model, test);
    printf("Result: %f\n", pred); // Expected: ~12.0
    
    // 4. Free
    linreg_free(model);
    return 0;
}
```

### Logistic Regression
```c
#include <stdio.h>
#include "logistic_reg.h"

int main() {
    // Binary Classification Data
    double X[] = { 1, 2, 8, 9 }; // Low vs High
    double y[] = { 0, 0, 1, 1 }; 
    
    RegressionConfig config = { .learning_rate = 0.1, .num_iterations = 5000, .early_stopping_threshold = 1e-6 };
    
    // 1. Create
    RegressionModel *model = logreg_create(1);
    
    // 2. Train
    logreg_train(model, X, y, 4, &config);
    
    // 3. Predict Probability
    double test[] = { 2.5 };
    double prob = logreg_predict(model, test);
    printf("Prob: %f\n", prob); // Expected: ~0.0
    
    // 4. Free
    logreg_free(model);
    return 0;
}
```

---

## 🧠 Design Philosophy

1. **Minimalism**: The library avoids defining complex tensor types. Standard C arrays (`double*`) are used for maximum compatibility and ease of integration with other systems.
2. **Transparency**: The API clearly separates creation, configuration, training, and prediction, allowing the user full control over the lifecycle.
3. **No External Dependencies**: The library relies solely on the C standard library (`math.h`, `stdlib.h`, `stdio.h`), ensuring it is easy to port and compile anywhere.
