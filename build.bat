@echo off
if not exist lib mkdir lib

echo Building Library...
gcc -c src/linear_reg.c -Iinclude -o linear_reg.o
gcc -c src/logistic_reg.c -Iinclude -o logistic_reg.o
ar rcs lib/libmllib.a linear_reg.o logistic_reg.o
del *.o

echo Building Examples...
gcc examples/linear_regression_example.c -Iinclude -Llib -lmllib -o examples/linear_regression_example.exe
gcc examples/logistic_regression_example.c -Iinclude -Llib -lmllib -o examples/logistic_regression_example.exe

echo Build Complete!
