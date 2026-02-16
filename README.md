# **mllib – Machine Learning Library in C**

**mllib** is a lightweight and extensible **Machine Learning Library** for C. It provides fundamental ML models like **Linear Regression** and **Logistic Regression**, with plans to expand to other models in the future.

## 🚀 **Features**
✅ Simple and efficient ML algorithms in pure C.
✅ Lightweight, with no external dependencies.
✅ Easy-to-use API with `#include "mllib.h"`.
✅ Open-source and extensible for future models.

---

## 📥 **Installation**
Clone the repository and build the project using `build.bat`:
```sh
git clone https://github.com/Sushrutha05/ML-C-Library.git
cd ML-C-Library
build.bat
```

Then, include the main header in your code:
```c
#include "mllib.h"
```

---

## 📖 **Usage**

### **Linear Regression**
Use `lin_reg()` to perform simple linear regression.

#### **Example**
```c
#include <stdio.h>
#include "mllib.h"

int main() {
    double x_arr[] = {1, 2, 3, 4, 5};
    double y_arr[] = {2, 4, 6, 8, 10};
    
    // Predict for x = 6
    double predicted = lin_reg(x_arr, y_arr, 5, 6, "double");
    printf("Predicted value: %.2lf\n", predicted);

    return 0;
}
```

### **Logistic Regression**
Use `logreg_create()`, `logreg_train()`, and `logreg_predict()` for binary classification.

#### **Example**
```c
#include <stdio.h>
#include "mllib.h"

int main() {
    // ... setup data ...
    LogRegModel *model = logreg_create(num_features);
    logreg_train(model, X, y, num_samples, config);
    
    int class_label = logreg_predict(model, sample, 0.5);
    logreg_free(model);
    return 0;
}
```

Check `examples/` for full working code.

---

## 🛠 **Building**
The project includes a `build.bat` script to automate the build process on Windows.

- **Build Library & Examples**:
  ```cmd
  build.bat
  ```
- **Clean Build Artifacts**:
  Manually delete `.o`, `.a`, `.exe` files or create a clean script.

---

## 📌 **Planned Models**
- [x] **Linear Regression**
- [x] **Logistic Regression**
- [ ] **K-Nearest Neighbors (KNN)**
- [ ] **Support Vector Machines (SVM)**
- [ ] **Decision Trees**
