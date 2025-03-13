# **mllib – Machine Learning Library in C**  

**mllib** is a lightweight and extensible **Machine Learning Library** for C. It provides fundamental ML models like **Linear Regression**, with plans to expand to other models in the future.  

## 🚀 **Features**  
✅ Simple and efficient ML algorithms in pure C.  
✅ Lightweight, with no external dependencies.  
✅ Easy-to-use API with `#include "mllib.h"`.  
✅ Open-source and extensible for future models.  

---

## 📥 **Installation**  
Clone the repository and include the required headers in your project:  
```sh
git clone https://github.com/Sushrutha05/ML-C-Library.git
```
Then, include the main header in your code:  
```c
#include "mllib.h"
```

---

## 📖 **Usage**  

### **Linear Regression**  
Use `lin_reg()` to perform simple linear regression.  

#### **Function Declaration**  
```c
double lin_reg(const void* x_arr, const void* y_arr, int no_ele, double x, const char* type);
```

#### **Example**  
```c
#include <stdio.h>
#include "mllib.h"

int main() {
    double x_arr[] = {1, 2, 3, 4, 5};
    double y_arr[] = {2, 4, 6, 8, 10};
    
    double predicted = lin_reg(x_arr, y_arr, 5, 6, "double");
    printf("Predicted value: %.2lf\n", predicted);

    return 0;
}
```

#### **Compiling and Running**  
```sh
gcc -o my_program my_program.c linear_reg.c -lm
./my_program
```

---

## 📌 **Planned Models**   
- [ ] **Logistic Regression**  
- [ ] **K-Nearest Neighbors (KNN)**  
- [ ] **Support Vector Machines (SVM)**  
- [ ] **Decision Trees**  
