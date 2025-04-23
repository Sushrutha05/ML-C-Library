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
If you plan to use the static library (libmllib.a), make sure it is built and available in your ```lib/ ```directory.

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
To compile and link your program with the static library (```libmllib.a```), use the following command:
```sh
gcc my_program.c -Iinclude -Llib -lmllib -o my_program
./my_program
```
This command does the following:

-Iinclude: Tells the compiler to look for header files in the include/ directory.

-Llib: Tells the linker to look for libraries in the lib/ directory.

-lmllib: Links the program with libmllib.a (the static library).

If you haven't already built the static library (libmllib.a), follow the instructions below to create it.

Building the Static Library (libmllib.a)
To build the static library (libmllib.a), run the following commands in the project directory:

1.Compile the source files into object files:
```
gcc -c src/linear_reg.c -Iinclude -o linear_reg.o
```
2. Create the static library:
```
ar rcs lib/libmllib.a linear_reg.o
```
Once the static library is created, you can use it in your programs as described above.
---

## 📌 **Planned Models**   
- [ ] **Logistic Regression**  
- [ ] **K-Nearest Neighbors (KNN)**  
- [ ] **Support Vector Machines (SVM)**  
- [ ] **Decision Trees**  
